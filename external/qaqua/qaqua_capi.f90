! A C interface to the q-AQUA v1 water potential (Q. Yu, C. Qu, P. L. Houston, R. Conte, A. Nandi, J. M. Bowman,
! J. Phys. Chem. Lett. 13, 5068 (2022); github.com/jmbowma/q-AQUA, MIT licence), compiled together with its sources
! by external/qaqua/build.sh.  Their default REAL is 8 bytes (ifort -r8; gfortran -fdefault-real-8 -fdefault-double-8),
! which is c_double.
!
! Conventions are q-AQUA's: positions in bohr in q-AQUA's atom order -- all hydrogens, two per molecule in molecule
! order, then all oxygens (H1a H2a H1b H2b ... Oa Ob ...) -- energies in hartree, gradients in hartree/bohr.  The
! total energy is the 4-body-truncated many-body expansion  sum 1b + sum 2b + sum 3b + sum 4b, with the 1-body term the
! Partridge-Schwenke monomer potential (zero at its minimum).
!
! pes_init reads the coefficients from '../coef/' relative to the working directory, which the caller sets (the
! Python driver changes into <q-AQUA>/src for the call), and allocates its work arrays for a fixed number of waters
! without checking whether they are allocated; it is therefore called once, and every later call with another
! number of waters re-sizes the work arrays here instead.
module qaqua_capi
  use iso_c_binding
  use pes_shell
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
    if (allocated(chg_gd)) deallocate(chg_gd)
    nw = n
    n4b = nw*(nw-1)*(nw-2)*(nw-3)/24
    n3b = nw*(nw-1)*(nw-2)/6
    n2b = nw*(nw-1)/2
    allocate(x2b(n2b,3,7), x3b(n3b,3,10), x4b(n4b,3,13))
    allocate(loc2b(n2b,2), loc3b(n3b,3), loc4b(n4b,4))
    allocate(chg_gd(nw,3,19))
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

  ! read the coefficients (working directory: <q-AQUA>/src) and size the work arrays for n waters; returns 0
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

  ! the 1-body energy of one monomer, x(9) = H1, H2, O in bohr: q-AQUA's pot1b (Partridge-Schwenke + its offset)
  subroutine qaqua_onebody(x, e) bind(C, name="qaqua_onebody")
    real(c_double), intent(in) :: x(9)
    real(c_double), intent(out) :: e
    real :: x1(3,3), p
    if (.not. ready) stop "qaqua_onebody: call qaqua_init first"
    x1 = reshape(x, [3,3])
    p = 0
    call pot1b(x1, p)
    e = p
  end subroutine qaqua_onebody

  ! the 1-body energy and gradient exactly as pot_gd computes them (pot_gd_1b: a central difference, h = 1e-3 bohr),
  ! so that the total gradient minus the sum of these is exactly the gradient of the n >= 2 body energy
  subroutine qaqua_onebody_gradient(x, e, g) bind(C, name="qaqua_onebody_gradient")
    real(c_double), intent(in) :: x(9)
    real(c_double), intent(out) :: e, g(9)
    real :: x1(3,3), p, g1(3,3)
    if (.not. ready) stop "qaqua_onebody_gradient: call qaqua_init first"
    x1 = reshape(x, [3,3])
    p = 0
    g1 = 0
    call pot_gd_1b(x1, p, g1)
    e = p
    g = reshape(g1, [9])
  end subroutine qaqua_onebody_gradient

  ! the LTP2011 point charges of one monomer (H1, H2, O; e) and its dipole (e bohr, the charges' sum q r): the charges
  ! q-AQUA's 2-body term switches to over 6.5-7.8 A and uses alone beyond
  subroutine qaqua_charges(x, q, d) bind(C, name="qaqua_charges")
    real(c_double), intent(in) :: x(9)
    real(c_double), intent(out) :: q(3), d(3)
    real :: x1(3,3), qq(3), dd(3)
    x1 = reshape(x, [3,3])
    call dip_ltp2011(x1, dd, qq)
    q = qq
    d = dd
  end subroutine qaqua_charges

end module qaqua_capi
