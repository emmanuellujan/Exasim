/*
  GMRESMPIR solver:
  Reuses GMRES in gmres.cpp and applies iterative refinement in an outer loop.
*/
#ifndef __GMRESMPIRSOLVER
#define __GMRESMPIRSOLVER 

Int GMRESMPIR(sysstruct &sys, CDiscretization &disc, CPreconditioner& prec, Int backend)
{
    INIT_TIMING;

    Int ncu = disc.common.ncu;
    Int npe = disc.common.npe;
    Int ne = disc.common.ne1;
    Int N = npe*ncu*ne;
    // Prefer allocated system-vector sizes to avoid accidental overruns.
    if (sys.szb > 0) N = sys.szb;
    if (sys.szx > 0) N = (N > sys.szx) ? sys.szx : N;
    if (sys.szr > 0) N = (N > sys.szr) ? sys.szr : N;
    if (N <= 0) {
        disc.common.linearSolverRelError = 0.0;
        return 0;
    }
    const Int maxitOuter = 1;
    Int j = 0;
    dstype nrmb, nrmr, relerr, relbest, tol;
    const dstype refineGate = (dstype)1.0e-2;
    const dstype acceptFactor = (dstype)0.9;

    // Save original RHS and working iterate.
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
    auto cleanup = [&]() {
        TemplateFree(b0, backend);
        TemplateFree(xk, backend);
        TemplateFree(xcand, backend);
        TemplateFree(rk, backend);
        TemplateFree(rhs, backend);
    };

    ArrayCopy(disc.common.cublasHandle, b0, sys.b, N, backend);
    nrmb = PNORM(disc.common.cublasHandle, N, b0, backend);
    if (nrmb < 1.0e-30) {
        ArraySetValue(sys.x, zero, N);
        disc.common.linearSolverRelError = 0.0;
        cleanup();
        return 0;
    }

    // x0 = GMRES(A, b, m)
    j += GMRES(sys, disc, prec, backend);
    ArrayCopy(disc.common.cublasHandle, xk, sys.x, N, backend);

    tol = disc.common.linearSolverTol;
    // True residual of baseline GMRES solution.
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

    Int naccepted = 0;
    for (Int k = 0; k < maxitOuter; k++) {
        // rk = -b - A*xk
        disc.evalMatVec(rk, xk, sys.u, b0, backend);
        ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
        nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
        disc.common.linearSolverRelError = relerr;
        if (relerr < tol)
            break;

        // dk = GMRES(A, rk, m): GMRES solves A*d = -rhs, so rhs = -rk
        ArrayAXPBY(rhs, rk, rk, minusone, zero, N);
        ArrayCopy(disc.common.cublasHandle, sys.b, rhs, N, backend);
        ArraySetValue(sys.x, zero, N);
        j += GMRES(sys, disc, prec, backend);

        // Candidate update xk+1 = xk + dk
        ArrayAXPBY(xcand, xk, sys.x, one, one, N);
        disc.evalMatVec(rk, xcand, sys.u, b0, backend);
        ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
        nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);

        // Accept only if true residual decreases; otherwise stop refinement.
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
                printf("GMRESMPIR rejects refinement step: old=%g new=%g (required new < %g*old)\n", relbest, relerr, acceptFactor);
            break;
        }
    }

    // Write final iterate and restore original RHS.
    ArrayCopy(disc.common.cublasHandle, sys.x, xk, N, backend);
    ArrayCopy(disc.common.cublasHandle, sys.b, b0, N, backend);

    // Final true residual for reporting.
    disc.evalMatVec(rk, sys.x, sys.u, b0, backend);
    ArrayAXPBY(rk, b0, rk, minusone, minusone, N);
    nrmr = PNORM(disc.common.cublasHandle, N, rk, backend);
    relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
    disc.common.linearSolverRelError = relerr;

    if (relerr > tol && disc.common.mpiRank==0) {
        printf("Warning: GMRESMPIR does not converge to the tolerance %g within %d refinement steps\n", tol, maxitOuter);
        printf("Warning: The current relative error is %g \n", relerr);
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
    // Guard against inconsistent sizes between caller and allocated system vectors.
    if (sys.szb > 0 && Nwork > sys.szb) Nwork = sys.szb;
    if (sys.szx > 0 && Nwork > sys.szx) Nwork = sys.szx;
    if (sys.szr > 0 && Nwork > sys.szr) Nwork = sys.szr;
    if (Nwork <= 0) {
        disc.common.linearSolverRelError = 0.0;
        return 0;
    }

    if (Nwork != N && disc.common.mpiRank==0) {
        printf("Warning: GMRESMPIR received N=%d but using N=%d to match allocated buffers.\n", N, Nwork);
    }

    const Int maxitOuter = 1;
    Int j = 0;
    dstype nrmb, nrmr, relerr, relbest, tol;
    dstype alpha = spatialScheme == 0 ? minusone : one;
    const dstype refineGate = (dstype)1.0e-2;
    const dstype acceptFactor = (dstype)0.9;

    // Save original RHS and working iterate.
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
    auto cleanup = [&]() {
        TemplateFree(b0, backend);
        TemplateFree(xk, backend);
        TemplateFree(xcand, backend);
        TemplateFree(rk, backend);
        TemplateFree(rhs, backend);
    };

    ArrayCopy(disc.common.cublasHandle, b0, sys.b, Nwork, backend);
    nrmb = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, b0, backend);
    if (nrmb < disc.common.nonlinearSolverTol) {
        ArraySetValue(sys.x, zero, Nwork);
        disc.common.linearSolverRelError = 0.0;
        cleanup();
        return 0;
    }

    // x0 = GMRES(A, b, m)
    j += GMRES(sys, disc, prec, Nwork, spatialScheme, backend);
    ArrayCopy(disc.common.cublasHandle, xk, sys.x, Nwork, backend);

    tol = disc.common.linearSolverTol;
    // True residual of baseline GMRES solution.
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

    Int naccepted = 0;
    for (Int k = 0; k < maxitOuter; k++) {
        // rk = alpha*b - A*xk
        disc.evalMatVec(rk, xk, sys.u, b0, spatialScheme, backend);
        ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
        nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
        disc.common.linearSolverRelError = relerr;
        if (relerr < tol)
            break;

        // dk = GMRES(A, rk, m): GMRES solves A*d = alpha*rhs, so rhs = alpha*rk
        ArrayAXPBY(rhs, rk, rk, alpha, zero, Nwork);
        ArrayCopy(disc.common.cublasHandle, sys.b, rhs, Nwork, backend);
        ArraySetValue(sys.x, zero, Nwork);
        j += GMRES(sys, disc, prec, Nwork, spatialScheme, backend);

        // Candidate update xk+1 = xk + dk
        ArrayAXPBY(xcand, xk, sys.x, one, one, Nwork);
        disc.evalMatVec(rk, xcand, sys.u, b0, spatialScheme, backend);
        ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
        nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
        relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);

        // Accept only if true residual decreases; otherwise stop refinement.
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
                printf("GMRESMPIR rejects refinement step: old=%g new=%g (required new < %g*old)\n", relbest, relerr, acceptFactor);
            break;
        }
    }

    // Write final iterate and restore original RHS.
    ArrayCopy(disc.common.cublasHandle, sys.x, xk, Nwork, backend);
    ArrayCopy(disc.common.cublasHandle, sys.b, b0, Nwork, backend);

    // Final true residual for reporting.
    disc.evalMatVec(rk, sys.x, sys.u, b0, spatialScheme, backend);
    ArrayAXPBY(rk, b0, rk, alpha, minusone, Nwork);
    nrmr = PNORM(disc.common.cublasHandle, Nwork, disc.common.ndofuhatinterface, rk, backend);
    relerr = nrmr/std::max(nrmb, (dstype)1.0e-30);
    disc.common.linearSolverRelError = relerr;

    if (relerr > tol && disc.common.mpiRank==0) {
        printf("Warning: GMRESMPIR does not converge to the tolerance %g within %d refinement steps\n", tol, maxitOuter);
        printf("Warning: The current relative error is %g \n", relerr);
        if (naccepted == 0)
            printf("Warning: GMRESMPIR accepted 0 refinement steps and returned baseline GMRES solution.\n");
    }

    cleanup();
    return j;
}

#endif
