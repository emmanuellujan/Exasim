/*
  GMRESMPIR solver:
  Reuses GMRES in gmres.cpp and applies iterative refinement in an outer loop.

  Note: true mixed precision would require a native low-precision GMRES path.
  Here we emulate low-precision solves by quantizing GMRES inputs/outputs to float.
*/
#ifndef __GMRESMPIRSOLVER
#define __GMRESMPIRSOLVER

static inline void QuantizeToLowPrecision(dstype *v, Int N, Int backend,
                                          dstype *hostHi,
                                          dstype_low *hostLo)
{
#ifdef USE_FLOAT
    (void)v; (void)N; (void)backend; (void)hostHi; (void)hostLo;
    return;
#else
    if (N <= 0 || v == nullptr) return;

    if (backend <= 1) {
        for (Int i = 0; i < N; i++)
            v[i] = (dstype)((dstype_low)v[i]);
        return;
    }

    if (hostHi == nullptr || hostLo == nullptr)
        return;

    TemplateCopytoHost(hostHi, v, N, backend);
    for (Int i = 0; i < N; i++)
        hostLo[i] = (dstype_low)hostHi[i];
    for (Int i = 0; i < N; i++)
        hostHi[i] = (dstype)hostLo[i];
    TemplateCopytoDevice(v, hostHi, N, backend);
#endif
}

Int GMRESMPIR(sysstruct &sys, CDiscretization &disc, CPreconditioner& prec, Int backend)
{
    INIT_TIMING;

    Int ncu = disc.common.ncu;
    Int npe = disc.common.npe;
    Int ne = disc.common.ne1;
    Int N = npe*ncu*ne;

    if (sys.szb > 0) N = sys.szb;
    if (sys.szx > 0) N = (N > sys.szx) ? sys.szx : N;
    if (sys.szr > 0) N = (N > sys.szr) ? sys.szr : N;
    if (N <= 0) {
        disc.common.linearSolverRelError = 0.0;
        return 0;
    }

    // IR tuning: small outer loop, strict acceptance, and guarded entry.
    const Int maxitOuter = 4;
    const dstype refineGate = (dstype)2.5e-2;
    const dstype acceptFactor = (dstype)0.95;
    const Int correctionIterBoost = 2;

    Int j = 0;
    Int naccepted = 0;
    Int maxitOrig = disc.common.linearSolverMaxIter;
    dstype nrmb, nrmr, relerr, relbest, tol;

    dstype *b0 = nullptr;
    dstype *xk = nullptr;
    dstype *xcand = nullptr;
    dstype *rk = nullptr;
    dstype *rhs = nullptr;
    TemplateMalloc(&b0, N, backend);
    TemplateMalloc(&xk, N, backend);
    TemplateMalloc(&xcand, N, backend);
    TemplateMalloc(&rk, N, backend);
    TemplateMalloc(&rhs, N, backend);

    dstype *hostHi = nullptr;
    dstype_low *hostLo = nullptr;
#ifndef USE_FLOAT
    if (backend > 1) {
        TemplateMalloc(&hostHi, N, 0);
        TemplateMalloc(&hostLo, N, 0);
    }
#endif

    auto cleanup = [&]() {
        disc.common.linearSolverMaxIter = maxitOrig;
        TemplateFree(b0, backend);
        TemplateFree(xk, backend);
        TemplateFree(xcand, backend);
        TemplateFree(rk, backend);
        TemplateFree(rhs, backend);
#ifndef USE_FLOAT
        TemplateFree(hostHi, 0);
        TemplateFree(hostLo, 0);
#endif
    };

    ArrayCopy(disc.common.cublasHandle, b0, sys.b, N, backend);
    nrmb = PNORM(disc.common.cublasHandle, N, b0, backend);
    if (nrmb < (dstype)1.0e-30) {
        ArraySetValue(sys.x, zero, N);
        disc.common.linearSolverRelError = 0.0;
        ArrayCopy(disc.common.cublasHandle, sys.b, b0, N, backend);
        cleanup();
        return 0;
    }

    if (disc.common.mpiRank==0)
        printf("GMRESMPIR: low-precision solve emulation active (float quantization).\n");

    // Step 1: x0 = GMRES(A,b,...) in emulated low precision.
    QuantizeToLowPrecision(sys.b, N, backend, hostHi, hostLo);
    QuantizeToLowPrecision(sys.x, N, backend, hostHi, hostLo);
    j += GMRES(sys, disc, prec, backend);
    QuantizeToLowPrecision(sys.x, N, backend, hostHi, hostLo);
    ArrayCopy(disc.common.cublasHandle, xk, sys.x, N, backend);

    tol = disc.common.linearSolverTol;

    // Step 2 (double): rk = b - A*xk (sign convention in this code path: -b - A*xk).
    disc.evalMatVec(rk, xk, sys.u, b0, backend);
    ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
    nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
    relbest = nrmr/std::max(nrmb, (dstype)1.0e-30);
    disc.common.linearSolverRelError = relbest;

    if (disc.common.mpiRank==0)
        printf("GMRESMPIR baseline true relative error: %g\n", relbest);

    if (relbest < tol) {
        ArrayCopy(disc.common.cublasHandle, sys.b, b0, N, backend);
        cleanup();
        return j;
    }

    if (relbest > refineGate) {
        if (disc.common.mpiRank==0)
            printf("GMRESMPIR skips refinement because baseline error %g is above gate %g.\n", relbest, refineGate);
        ArrayCopy(disc.common.cublasHandle, sys.b, b0, N, backend);
        cleanup();
        return j;
    }

    // Steps 3-5: IR loop.
    for (Int k = 0; k < maxitOuter; k++) {
        disc.evalMatVec(rk, xk, sys.u, b0, backend);
        ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
        nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
        disc.common.linearSolverRelError = relerr;

        if (disc.common.mpiRank==0)
            printf("GMRESMPIR step %d residual: %g\n", (int)(k+1), relerr);

        if (relerr < tol)
            break;

        // Step 4: dk = GMRES(A,rk,...) in emulated low precision.
        ArrayAXPBY(rhs, rk, rk, minusone, zero, N);
        ArrayCopy(disc.common.cublasHandle, sys.b, rhs, N, backend);
        QuantizeToLowPrecision(sys.b, N, backend, hostHi, hostLo);
        ArraySetValue(sys.x, zero, N);

        disc.common.linearSolverMaxIter = std::max(maxitOrig, correctionIterBoost*maxitOrig);
        j += GMRES(sys, disc, prec, backend);
        disc.common.linearSolverMaxIter = maxitOrig;

        QuantizeToLowPrecision(sys.x, N, backend, hostHi, hostLo);

        // Step 5 (double): x_{k+1} = x_k + d_k, then evaluate true residual.
        ArrayAXPBY(xcand, xk, sys.x, one, one, N);
        disc.evalMatVec(rk, xcand, sys.u, b0, backend);
        ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
        nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);

        if (relerr < acceptFactor*relbest) {
            ArrayCopy(disc.common.cublasHandle, xk, xcand, N, backend);
            relbest = relerr;
            naccepted += 1;
            disc.common.linearSolverRelError = relbest;
            if (relbest < tol)
                break;
        }
        else {
            if (disc.common.mpiRank==0)
                printf("GMRESMPIR rejects refinement step: old=%g new=%g (required new < %g*old)\n",
                       relbest, relerr, acceptFactor);
            break;
        }
    }

    ArrayCopy(disc.common.cublasHandle, sys.x, xk, N, backend);
    ArrayCopy(disc.common.cublasHandle, sys.b, b0, N, backend);

    disc.evalMatVec(rk, sys.x, sys.u, b0, backend);
    ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
    nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
    relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
    disc.common.linearSolverRelError = relerr;

    if (relerr > tol && disc.common.mpiRank==0) {
        printf("Warning: GMRESMPIR does not converge to the tolerance %g within %d refinement steps\n", tol, maxitOuter);
        printf("Warning: The current relative error is %g\n", relerr);
        if (naccepted == 0)
            printf("Warning: GMRESMPIR accepted 0 refinement steps and returned baseline GMRES solution.\n");
    }

    cleanup();
    return j;
}

Int GMRESMPIR(sysstruct &sys, CDiscretization &disc, CPreconditioner& prec, Int N, Int spatialScheme, Int backend)
{
    INIT_TIMING;

    Int Nwork = N;
    if (sys.szb > 0 && Nwork > sys.szb) Nwork = sys.szb;
    if (sys.szx > 0 && Nwork > sys.szx) Nwork = sys.szx;
    if (sys.szr > 0 && Nwork > sys.szr) Nwork = sys.szr;
    if (Nwork <= 0) {
        disc.common.linearSolverRelError = 0.0;
        return 0;
    }

    if (Nwork != N && disc.common.mpiRank==0)
        printf("Warning: GMRESMPIR received N=%d but using N=%d to match allocated buffers.\n", N, Nwork);

    const Int maxitOuter = 4;
    const dstype refineGate = (dstype)2.5e-2;
    const dstype acceptFactor = (dstype)0.95;
    const Int correctionIterBoost = 2;

    Int j = 0;
    Int naccepted = 0;
    Int maxitOrig = disc.common.linearSolverMaxIter;
    dstype nrmb, nrmr, relerr, relbest, tol;
    dstype alpha = spatialScheme == 0 ? minusone : one;

    dstype *b0 = nullptr;
    dstype *xk = nullptr;
    dstype *xcand = nullptr;
    dstype *rk = nullptr;
    dstype *rhs = nullptr;
    TemplateMalloc(&b0, Nwork, backend);
    TemplateMalloc(&xk, Nwork, backend);
    TemplateMalloc(&xcand, Nwork, backend);
    TemplateMalloc(&rk, Nwork, backend);
    TemplateMalloc(&rhs, Nwork, backend);

    dstype *hostHi = nullptr;
    dstype_low *hostLo = nullptr;
#ifndef USE_FLOAT
    if (backend > 1) {
        TemplateMalloc(&hostHi, Nwork, 0);
        TemplateMalloc(&hostLo, Nwork, 0);
    }
#endif

    auto cleanup = [&]() {
        disc.common.linearSolverMaxIter = maxitOrig;
        TemplateFree(b0, backend);
        TemplateFree(xk, backend);
        TemplateFree(xcand, backend);
        TemplateFree(rk, backend);
        TemplateFree(rhs, backend);
#ifndef USE_FLOAT
        TemplateFree(hostHi, 0);
        TemplateFree(hostLo, 0);
#endif
    };

    ArrayCopy(disc.common.cublasHandle, b0, sys.b, Nwork, backend);
    nrmb = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, b0, backend);
    if (nrmb < disc.common.nonlinearSolverTol) {
        ArraySetValue(sys.x, zero, Nwork);
        disc.common.linearSolverRelError = 0.0;
        ArrayCopy(disc.common.cublasHandle, sys.b, b0, Nwork, backend);
        cleanup();
        return 0;
    }

    if (disc.common.mpiRank==0)
        printf("GMRESMPIR: low-precision solve emulation active (float quantization).\n");

    // Step 1: x0 = GMRES(A,b,...) in emulated low precision.
    QuantizeToLowPrecision(sys.b, Nwork, backend, hostHi, hostLo);
    QuantizeToLowPrecision(sys.x, Nwork, backend, hostHi, hostLo);
    j += GMRES(sys, disc, prec, Nwork, spatialScheme, backend);
    QuantizeToLowPrecision(sys.x, Nwork, backend, hostHi, hostLo);
    ArrayCopy(disc.common.cublasHandle, xk, sys.x, Nwork, backend);

    tol = disc.common.linearSolverTol;

    // Step 2 (double): rk = b - A*xk (in this path alpha*b - A*xk).
    disc.evalMatVec(rk, xk, sys.u, b0, spatialScheme, backend);
    ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
    nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
    relbest = nrmr/std::max(nrmb, (dstype)1.0e-30);
    disc.common.linearSolverRelError = relbest;

    if (disc.common.mpiRank==0)
        printf("GMRESMPIR baseline true relative error: %g\n", relbest);

    if (relbest < tol) {
        ArrayCopy(disc.common.cublasHandle, sys.b, b0, Nwork, backend);
        cleanup();
        return j;
    }

    if (relbest > refineGate) {
        if (disc.common.mpiRank==0)
            printf("GMRESMPIR skips refinement because baseline error %g is above gate %g.\n", relbest, refineGate);
        ArrayCopy(disc.common.cublasHandle, sys.b, b0, Nwork, backend);
        cleanup();
        return j;
    }

    for (Int k = 0; k < maxitOuter; k++) {
        disc.evalMatVec(rk, xk, sys.u, b0, spatialScheme, backend);
        ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
        nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
        disc.common.linearSolverRelError = relerr;

        if (disc.common.mpiRank==0)
            printf("GMRESMPIR step %d residual: %g\n", (int)(k+1), relerr);

        if (relerr < tol)
            break;

        ArrayAXPBY(rhs, rk, rk, alpha, zero, Nwork);
        ArrayCopy(disc.common.cublasHandle, sys.b, rhs, Nwork, backend);
        QuantizeToLowPrecision(sys.b, Nwork, backend, hostHi, hostLo);
        ArraySetValue(sys.x, zero, Nwork);

        disc.common.linearSolverMaxIter = std::max(maxitOrig, correctionIterBoost*maxitOrig);
        j += GMRES(sys, disc, prec, Nwork, spatialScheme, backend);
        disc.common.linearSolverMaxIter = maxitOrig;

        QuantizeToLowPrecision(sys.x, Nwork, backend, hostHi, hostLo);

        ArrayAXPBY(xcand, xk, sys.x, one, one, Nwork);
        disc.evalMatVec(rk, xcand, sys.u, b0, spatialScheme, backend);
        ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
        nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);

        if (relerr < acceptFactor*relbest) {
            ArrayCopy(disc.common.cublasHandle, xk, xcand, Nwork, backend);
            relbest = relerr;
            naccepted += 1;
            disc.common.linearSolverRelError = relbest;
            if (relbest < tol)
                break;
        }
        else {
            if (disc.common.mpiRank==0)
                printf("GMRESMPIR rejects refinement step: old=%g new=%g (required new < %g*old)\n",
                       relbest, relerr, acceptFactor);
            break;
        }
    }

    ArrayCopy(disc.common.cublasHandle, sys.x, xk, Nwork, backend);
    ArrayCopy(disc.common.cublasHandle, sys.b, b0, Nwork, backend);

    disc.evalMatVec(rk, sys.x, sys.u, b0, spatialScheme, backend);
    ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
    nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
    relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
    disc.common.linearSolverRelError = relerr;

    if (relerr > tol && disc.common.mpiRank==0) {
        printf("Warning: GMRESMPIR does not converge to the tolerance %g within %d refinement steps\n", tol, maxitOuter);
        printf("Warning: The current relative error is %g\n", relerr);
        if (naccepted == 0)
            printf("Warning: GMRESMPIR accepted 0 refinement steps and returned baseline GMRES solution.\n");
    }

    cleanup();
    return j;
}

#endif
