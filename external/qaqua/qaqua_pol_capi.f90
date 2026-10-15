! A C interface to the q-AQUA-pol water potential (Q. Yu, C. Qu, P. L. Houston, A. Nandi, R. Conte, J. M. Bowman,
! "Interfacing q-AQUA with a polarizable force field: the best of both worlds", J. Chem. Theory Comput. 19, 3446
! (2023); github.com/jmbowma/q-AQUA-pol, CC0), compiled together with its sources by external/qaqua/build_pol.sh.
! Their default REAL is 8 bytes (ifort -r8; gfortran -fdefault-real-8 -fdefault-double-8), which is c_double.
!
! The potential is TTM3-F everywhere -- the Partridge-Schwenke monomer, geometry-dependent charges on the H atoms and
! the M-site, one Thole-smeared polarisable site per molecule (the M-site, 1.444 A^3) whose induced dipoles are
! iterated to self-consistency without cutoff, an O-O C6/R^6 + exponential term -- plus CCSD(T) corrections fitted as
! permutationally invariant polynomials: 2-body and 3-body switched off over 6.5-7.0 A (O-O, the largest O-O),
! 4-body over 5.5-6.5 A.
!
! The exported names and signatures are those of qaqua_capi.f90 (q-AQUA v1), so that sseft/qaqua.py drives both;
! conventions are q-AQUA's: positions in bohr in q-AQUA's atom order (all hydrogens, two per molecule in molecule
! order, then all oxygens), energies in hartree (TTM3-F's kcal/mol divided by the potential's own aukcal = 627.51),
! gradients in hartree/bohr.  The 1-body energy is the Partridge-Schwenke term TTM3-F adds per molecule (pot_nasa),
! so that the total minus the sum of these is the potential's interaction energy; the monomer charges are the
! Partridge-Schwenke dipole-surface charges on O, H, H (dms_nasa) from which TTM3-F derives its site charges (the
! same dipole; the M-site arrangement has another quadrupole).  qaqua_ttm returns TTM3-F's pieces and its induced
! dipoles for a configuration.
!
! pes_init reads the coefficients from '../coef_qAQUA-pol/' relative to the working directory, which the caller sets
! (the Python driver changes into <q-AQUA-pol>/src for the call), and allocates its work arrays for a fixed number of
! waters; it is called once, and every later call with another number of waters re-sizes them here instead (TTM3-F
! re-sizes its own arrays when the number of waters changes).
module qaqua_pol_capi
  use iso_c_binding
  use pes_shell, only: nw, n2b, n3b, n4b, x2b, x3b, x4b, loc2b, loc3b, loc4b, pes_init, getpot, pot_gd
  use constants, only: auang, aukcal
  use potential_mod, only: potential
  use model_mod, only: vdwC, vdwD, vdwE, polarM, CHARGECON
  use ttm3f_mod, only: init_ttm3f, t3_dip => dip, t3_efq => Efq, t3_charge => charge, t3_rm => RM, &
                       t3_fH => fH, t3_fM => fM, t3_fM3 => fM3
  implicit none
  logical, save :: ready = .false.

contains

  subroutine resize(n)
    integer, intent(in) :: n
    if (allocated(x2b)) deallocate(x2b)
    if (allocated(x3b)) deallocate(x3b)
    if (allocated(x4b)) deallocate(x4b)
    if (allocated(loc2b)) deallocate(loc2b)
    if (allocated(loc3b)) deallocate(loc3b)
    if (allocated(loc4b)) deallocate(loc4b)
    nw = n
    n4b = nw*(nw-1)*(nw-2)*(nw-3)/24
    n3b = nw*(nw-1)*(nw-2)/6
    n2b = nw*(nw-1)/2
    allocate(x2b(n2b,3,7), x3b(n3b,3,10), x4b(n4b,3,13))
    allocate(loc2b(n2b,2), loc3b(n3b,3), loc4b(n4b,4))
  end subroutine resize

  subroutine ensure(n)
    integer, intent(in) :: n
    if (.not. ready) then
      call pes_init(n)
      ready = .true.
    else if (n /= nw) then
      call resize(n)
    end if
  end subroutine ensure

  ! x(9n) in bohr, q-AQUA order -> (3, 3n) in A, TTM3-F order (all oxygens, then the hydrogens two per molecule)
  subroutine ttm_order(n, x, r)
    integer, intent(in) :: n
    real, intent(in) :: x(9*n)
    real, intent(out) :: r(3, 3*n)
    real :: xn(3, 3*n)
    xn = reshape(x, [3, 3*n])
    r(:, 1:n) = xn(:, 2*n+1:3*n)
    r(:, n+1:3*n) = xn(:, 1:2*n)
    r = r*auang
  end subroutine ttm_order

  ! x(9) = H1, H2, O in bohr -> (3, 3) O, H1, H2 in A
  subroutine monomer_ttm(x, r1)
    real, intent(in) :: x(9)
    real, intent(out) :: r1(3, 3)
    real :: xm(3, 3)
    xm = reshape(x, [3, 3])
    r1(:, 1) = xm(:, 3); r1(:, 2) = xm(:, 1); r1(:, 3) = xm(:, 2)
    r1 = r1*auang
  end subroutine monomer_ttm

  ! read the coefficients (working directory: <q-AQUA-pol>/src) and size the work arrays for n waters; returns 0
  integer(c_int) function qaqua_init(n) bind(C, name="qaqua_init")
    integer(c_int), value :: n
    call ensure(int(n))
    qaqua_init = 0
  end function qaqua_init

  ! 1 once the coefficients are read
  integer(c_int) function qaqua_ready() bind(C, name="qaqua_ready")
    qaqua_ready = merge(1, 0, ready)
  end function qaqua_ready

  ! total energy of n waters, x(9n) in bohr (q-AQUA order)
  subroutine qaqua_energy(n, x, e) bind(C, name="qaqua_energy")
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(9*n)
    real(c_double), intent(out) :: e
    real :: xx(9*n), ee
    call ensure(int(n))
    xx = x
    ee = 0
    call getpot(xx, ee)
    e = ee
  end subroutine qaqua_energy

  ! total energy and its gradient g(9n) (hartree/bohr, q-AQUA order)
  subroutine qaqua_energy_gradient(n, x, e, g) bind(C, name="qaqua_energy_gradient")
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(9*n)
    real(c_double), intent(out) :: e, g(9*n)
    real :: xx(9*n), ee, gg(3, 3*n)
    call ensure(int(n))
    xx = x
    ee = 0
    gg = 0
    call pot_gd(xx, ee, gg)
    e = ee
    g = reshape(gg, [9*n])
  end subroutine qaqua_energy_gradient

  ! the 1-body energy of one monomer, x(9) = H1, H2, O in bohr: the Partridge-Schwenke term TTM3-F adds per molecule
  subroutine qaqua_onebody(x, e) bind(C, name="qaqua_onebody")
    real(c_double), intent(in) :: x(9)
    real(c_double), intent(out) :: e
    real :: r1(3, 3), dr1(3, 3), e1
    call monomer_ttm(x, r1)
    call pot_nasa(r1, dr1, e1)
    e = e1/aukcal
  end subroutine qaqua_onebody

  ! the 1-body energy and its (analytic) gradient g(9) = H1, H2, O in hartree/bohr, as pot_gd adds them
  subroutine qaqua_onebody_gradient(x, e, g) bind(C, name="qaqua_onebody_gradient")
    real(c_double), intent(in) :: x(9)
    real(c_double), intent(out) :: e, g(9)
    real :: r1(3, 3), dr1(3, 3), e1, gm(3, 3)
    call monomer_ttm(x, r1)
    call pot_nasa(r1, dr1, e1)
    e = e1/aukcal
    gm(:, 1) = dr1(:, 2); gm(:, 2) = dr1(:, 3); gm(:, 3) = dr1(:, 1)
    g = reshape(gm/aukcal*auang, [9])
  end subroutine qaqua_onebody_gradient

  ! the Partridge-Schwenke dipole-surface charges of one monomer (H1, H2, O; e) and its dipole (e bohr): the charges
  ! TTM3-F moves to the H atoms and the M-site, keeping this dipole
  subroutine qaqua_charges(x, q, d) bind(C, name="qaqua_charges")
    real(c_double), intent(in) :: x(9)
    real(c_double), intent(out) :: q(3), d(3)
    real :: r1(3, 3), q3(3), dq3(3, 3, 3), xm(3, 3)
    call init_ttm3f(1)                                    ! sets the dipole-surface parameters dms_nasa reads
    call monomer_ttm(x, r1)
    call dms_nasa(r1, q3, dq3)
    q(1) = q3(2); q(2) = q3(3); q(3) = q3(1)
    xm = reshape(x, [3, 3])
    d = q(1)*xm(:, 1) + q(2)*xm(:, 2) + q(3)*xm(:, 3)
  end subroutine qaqua_charges

  ! TTM3-F alone for n waters, x(9n) in bohr (q-AQUA order): comp(5) = its total, the Partridge-Schwenke sum, the
  ! electrostatics of the site charges, the induction (-1/2 sum mu.E_q at self-consistency) and the O-O C6 +
  ! exponential term (hartree); mu(3n) its induced dipoles on the M-sites (e bohr), rm(3n) the M-sites (bohr) and
  ! qs(3n) the site charges H1, H2, M of every molecule (e)
  subroutine qaqua_ttm(n, x, comp, mu, rm, qs) bind(C, name="qaqua_ttm")
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(9*n)
    real(c_double), intent(out) :: comp(5), mu(3*n), rm(3*n), qs(3*n)
    real :: r(3, 3*n), dr(3, 3*n), en, eint, evdw, eind, eel, e1, dr1(3, 3), rij(3), d
    integer :: i, j, nn
    if (.not. ready) stop "qaqua_ttm: call qaqua_init first (it selects TTM3-F)"
    nn = int(n)
    call ttm_order(nn, x, r)
    call potential(nn, r, dr, en)
    eind = -0.5d0*sum(t3_dip(t3_fM3:t3_fM3+3*nn-1)*t3_efq(t3_fM3:t3_fM3+3*nn-1))
    eint = 0
    do i = 1, nn
      call pot_nasa(r(:, [i, nn+2*i-1, nn+2*i]), dr1, e1)
      eint = eint + e1
    end do
    evdw = 0
    do i = 1, nn - 1
      do j = i + 1, nn
        rij = r(:, i) - r(:, j)
        d = sqrt(sum(rij*rij))
        evdw = evdw + vdwC/d**6 + vdwD*exp(-vdwE*d)
      end do
    end do
    eel = en - eint - evdw - eind
    comp = [en, eint, eel, eind, evdw]/aukcal
    mu = t3_dip(t3_fM3:t3_fM3+3*nn-1)/CHARGECON/auang
    rm = reshape(t3_rm(:, t3_fM:t3_fM+nn-1), [3*nn])/auang
    do i = 1, nn
      qs(3*i-2) = t3_charge(t3_fH+2*i-2)/CHARGECON
      qs(3*i-1) = t3_charge(t3_fH+2*i-1)/CHARGECON
      qs(3*i) = t3_charge(t3_fM+i-1)/CHARGECON
    end do
  end subroutine qaqua_ttm

  ! TTM3-F's static polarisability (A^3, on the M-site) and its O-O dispersion coefficient C6 (kcal/mol A^6, the
  ! magnitude of the -C6/R^6 term)
  subroutine qaqua_ttm_constants(alpha, c6) bind(C, name="qaqua_ttm_constants")
    real(c_double), intent(out) :: alpha, c6
    call init_ttm3f(1)
    alpha = polarM
    c6 = -vdwC
  end subroutine qaqua_ttm_constants

end module qaqua_pol_capi
