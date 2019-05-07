from .veloxchemlib import MolecularOrbitals
from .veloxchemlib import Molecule
from .veloxchemlib import MolecularBasis
from .veloxchemlib import ExcitationVector
from .veloxchemlib import ElectricDipoleIntegralsDriver
from .veloxchemlib import ElectronRepulsionIntegralsDriver
from .veloxchemlib import OverlapIntegralsDriver
from .veloxchemlib import KineticEnergyIntegralsDriver
from .veloxchemlib import NuclearPotentialIntegralsDriver
from .veloxchemlib import mpi_master
from .veloxchemlib import szblock
from .veloxchemlib import denmat, fockmat
from .veloxchemlib import ericut

from .aofockmatrix import AOFockMatrix
from .aodensitymatrix import AODensityMatrix
from .outputstream import OutputStream

from itertools import product

from mpi4py import MPI

import numpy as np
import pandas as pd

class LinearResponse:
    """ Provides functionality to solve real linear
    response equations. """


    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._fock = None

    def get_orbital_numbers(self, mol_orbs, task):
        nocc = task.molecule.number_of_electrons() // 2
        norb = mol_orbs.number_mos()
        xv = ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True)
        cre = xv.bra_indexes()
        ann = xv.ket_indexes()
        return nocc, norb, xv, cre, ann

    def mat2vec(self, mat, mol_orbs, task):
        nocc, norb, xv, cre, ann = self.get_orbital_numbers(mol_orbs, task)
        vec_exc = [mat[k, l] for k, l in zip(cre, ann)]
        vec_dex = [mat[l, k] for k, l in zip(cre, ann)]
        return np.append(vec_exc, vec_dex)

    def paired(self, v_xy):
        """ returns paired trial vector """
        v_yx = v_xy.copy()
        half_rows = v_xy.shape[0]//2
        v_yx[:half_rows] = v_xy[half_rows:]
        v_yx[half_rows:] = v_xy[:half_rows]
        return v_yx

    def rm_lin_depend(self, basis, threshold=1e-10):
        """ removes linear dependencies from input basis vectors """
        Sb = basis.T@basis
        l, T = np.linalg.eig(Sb)
        b_norm = np.sqrt(Sb.diagonal())
        mask = l > threshold*b_norm
        return T[:, mask]

    def normalize(self, matrix):
        """ returns normalized and diagonalized matrix """
        eigvals, eigvecs = np.linalg.eig(matrix)
        Sdiag = np.diagonal(np.linalg.inv(eigvecs)@matrix@eigvecs)
        S12diag = Sdiag**-.5
        S12 = np.zeros((len(S12diag), len(S12diag)))
        np.fill_diagonal(S12, S12diag)
        return S12

    def get_dipole(self, task):
        dipoles = ElectricDipoleIntegralsDriver(self.rank, self.size, self.comm)
#        master_node = (self.rank == mpi_master())
#        if master_node:
#            mol = Molecule.read_xyz(self.xyz)
#            bas = MolecularBasis.read(mol, self.basis)
#        else:
#            mol = Molecule()
#            bas = MolecularBasis()
        D = dipoles.compute(task.molecule, task.ao_basis, self.comm)
        return D.x_to_numpy(), D.y_to_numpy(), D.z_to_numpy()

    def get_overlap(self, task):
        overlap = OverlapIntegralsDriver(self.rank, self.size, self.comm)
        S = overlap.compute(task.molecule, task.ao_basis, self.comm)
        return S.to_numpy()

    def construct_ed(self, mol_orbs, task):
        nocc, norb, xv, cre, ann = self.get_orbital_numbers(mol_orbs, task)
        ediag = 2*xv.diagonal_to_numpy(mol_orbs)
        return np.append(ediag, ediag)

    def construct_sd(self, mol_orbs, task):
        #to be reviewed!
        nocc, norb, xv, cre, ann = self.get_orbital_numbers(mol_orbs, task)
        sdiag = 2*np.ones((2, nocc*(norb-nocc)))
        sdiag[1, :] *= -1
        return sdiag.flatten()

    def get_rhs(self, mol_orbs, task, ops):
        """ create right-hand sides of linear response equations """
        if 'x' in ops or 'y' in ops or 'z' in ops:
            prop = {k: v for k, v in zip('xyz', self.get_dipole(task))}
        den = mol_orbs.get_density(task.molecule)
        da = den.alpha_to_numpy(0)
        db = den.beta_to_numpy(0)
        D = da + db
        S = self.get_overlap(task)
        mo = mol_orbs.alpha_to_numpy()

        matrices = tuple([mo.T@(S@D@prop[p].T - prop[p].T@D@S)@mo for p in ops])
        gradients = tuple([self.mat2vec(m, mol_orbs, task) for m in matrices])
        return gradients

    def initial_guess(self, mol_orbs, task, ops, freqs):
        # constructs diagonal elements of E[2]_0
        ed = self.construct_ed(mol_orbs, task)
        # constructs diagonal elements of S[2]
        sd = self.construct_sd(mol_orbs, task)
        ig = pd.DataFrame()
        for op, grad in zip(ops, self.get_rhs(mol_orbs, task, ops)):
            gn = np.linalg.norm(grad)
            for w in freqs:
                if gn < 1e-10:
                    ig[(op, w)] = np.zeros(ed.shape[0])
                else:
                    denom = ed - w*sd
                    ig[(op, w)] = grad/denom
        return ig

    def setup_trials(self, vectors, td=None, b=None, threshold=1e-10, normalize=True):
        trials = []
        for (op, freq) in vectors:
            vec = vectors[(op, freq)].values
            # preconditioning trials:
            if td is not None:
                v = vec/td[freq]
            else:
                v = vec
            if np.linalg.norm(v) > threshold:
                trials.append(v)
                if freq > threshold:
                    trials.append(self.paired(v))
        new_trials = np.around(np.array(trials).T, 10)
        if b is not None:
            new_trials = new_trials - (b @ b.T @ new_trials)
        if trials and normalize:
            # removing linear dependencies
            t = self.rm_lin_depend(new_trials)
            truncated = new_trials@t
            #S12 = self.normalize(truncated.T@truncated)
            for i in range(len(truncated[0])):
                truncated[:,i] = truncated[:,i]/np.linalg.norm(truncated[:,i])
            new_trials = truncated
            #new_trials = truncated@S12
        return new_trials

    def lr_solve(self, mol_orbs, task, ops='z', freqs=(0,0.5,), maxit=50, threshold=1e-6):
        V1 = pd.DataFrame({op: v for op, v in zip(ops, self.get_rhs(mol_orbs, task, ops))})
        igs = pd.DataFrame(self.initial_guess(mol_orbs, task, ops, freqs))
        b = self.setup_trials(igs)
        # projections of e2 and s2:
        e2b = self.e2n(b, mol_orbs, task)
        s2b = self.s2n(b, mol_orbs, task)
        od = self.construct_ed(mol_orbs, task)
        sd = self.construct_sd(mol_orbs, task)

        td = {w: od - w*sd for w in freqs}

        solutions = pd.DataFrame()
        residuals = pd.DataFrame()
        e2nn = pd.DataFrame()
        relative_residual_norm = pd.Series(index=igs.columns)

        for i in range(maxit):
            for op, freq in igs:
                v = np.around(V1[op].values, 10)
                c = np.linalg.inv(b.T @ (e2b - freq*s2b)) @ (b.T @ v)
                print(b.T @ e2b)
                solutions[(op, freq)] = b @ c
                e2nn[(op, freq)] = e2b @ c

#            e2nn = pd.DataFrame(self.e2n(solutions.values, mol_orbs, task), columns=solutions.columns)
            s2nn = pd.DataFrame(self.s2n(solutions.values, mol_orbs, task), columns=solutions.columns)

            for op, freq in igs:
                v = np.around(V1[op].values, 10)
                n = solutions[(op, freq)]
                r = e2nn[(op, freq)] - freq*s2nn[(op, freq)] - v
                residuals[(op, freq)] = r
                nv = np.dot(n, v)
                rn = np.linalg.norm(r)
                nn = np.linalg.norm(n)
                if nn != 0:
                    relative_residual_norm[(op, freq)] = rn / nn
                else:
                    relative_residual_norm[(op, freq)] = 0
                task.ostream.print_info(f"{i+1} <<{op};{op}>>({freq})={-nv:.10f} rn={rn:.5e} nn={nn:.5e}")
            task.ostream.print_blank()

            max_residual = max(relative_residual_norm)

            if max_residual < threshold:
                task.ostream.print_info('Converged')
                break
            new_trials = self.setup_trials(residuals, td=td, b=b)
            b = np.append(b, new_trials, axis=1)
            print(b.shape)
            new_e2b = self.e2n(new_trials, mol_orbs, task)
            new_s2b = self.s2n(new_trials, mol_orbs, task)
            e2b = np.append(e2b, new_e2b, axis=1)
            s2b = np.append(s2b, new_s2b, axis=1)
        return solutions

    def lr(self, mol_orbs, task):
        task.ostream.print_info('LINEAR RESPONSE OUTPUT:')
        task.ostream.print_blank()
        solutions = self.lr_solve(mol_orbs, task)
        task.ostream.print_info('Done!')

###############################################################################
###############################################################################
######################## COPIED PARTS FROM REAL SOLVER ########################
########################### and slightly adjusted #############################
###############################################################################
###############################################################################



    def get_two_el_fock(self, task, *dabs):

        mol = task.molecule
        bas = task.ao_basis

        dts = []
        for dab in dabs:
            da, db = dab
            dt = da + db
            ds = da - db
            dts.append(dt)
            dts.append(ds)
        dens = AODensityMatrix(dts, denmat.rest)
        fock = AOFockMatrix(dens)
        for i in range(0, 2*len(dabs), 2):
            fock.set_fock_type(fockmat.rgenjk, i)
            fock.set_fock_type(fockmat.rgenk, i+1)

        eri_driver = ElectronRepulsionIntegralsDriver(
            self.rank, self.size, self.comm
        )
        screening = eri_driver.compute(ericut.qqden, 1.0e-12, mol, bas)
        eri_driver.compute(ericut.qqden, 1.0e-12, mol, bas)
        eri_driver.compute(fock, dens, mol, bas, screening, self.comm)
        fock.reduce_sum(self.rank, self.size, self.comm)

        fabs = []
        for i in range(0, 2*len(dabs), 2):
            ft = fock.to_numpy(i).T
            fs = -fock.to_numpy(i+1).T

            fa = (ft + fs)/2
            fb = (ft - fs)/2

            fabs.append((fa, fb))

        return tuple(fabs)


    def get_one_el_hamiltonian(self, task):
        kinetic_driver = KineticEnergyIntegralsDriver(
            self.rank, self.size, self.comm
        )
        potential_driver = NuclearPotentialIntegralsDriver(
            self.rank, self.size, self.comm
        )

        mol = task.molecule
        bas = task.ao_basis

        T = kinetic_driver.compute(mol, bas, self.comm).to_numpy()
        V = potential_driver.compute(mol, bas, self.comm).to_numpy()

        return T-V



    def get_fock(self, mol_orbs, task):
        if self._fock is None:
            D = mol_orbs.get_density(task.molecule)
            da = D.alpha_to_numpy(0)
            db = D.beta_to_numpy(0)
            (fa, fb), = self.get_two_el_fock(task, (da, db),)
            h = self.get_one_el_hamiltonian(task)
            fa += h
            fb += h
            self._fock = (fa, fb)
        return self._fock


    def np2vlx(self, vec, mol_orbs, task):
        nocc, norb, xv, cre, ann = self.get_orbital_numbers(mol_orbs, task)
        zlen = len(vec) // 2
        z, y = vec[:zlen], vec[zlen:]
        xv.set_yzcoefficients(z, y)
        return xv


    def vec2mat(self, vec, mol_orbs, task):
        xv = self.np2vlx(vec, mol_orbs, task)
        kz = xv.get_zmatrix()
        ky = xv.get_ymatrix()

        rows = kz.number_of_rows() + ky.number_of_rows()
        cols = kz.number_of_columns() + ky.number_of_columns()

        kzy = np.zeros((rows, cols))
        kzy[:kz.number_of_rows(), ky.number_of_columns():] = kz.to_numpy()
        kzy[kz.number_of_rows():, :ky.number_of_columns()] = ky.to_numpy()

        return kzy


    def e2n(self, vecs, mol_orbs, task):
        vecs = np.array(vecs)

        S = self.get_overlap(task)
        D = mol_orbs.get_density(task.molecule)
        da = D.alpha_to_numpy(0)
        db = D.beta_to_numpy(0)
        fa, fb = self.get_fock(mol_orbs, task)
        mo = mol_orbs.alpha_to_numpy()

        if False:  # len(vecs.shape) == 1:


            kN = self.vec2mat(vecs, mol_orbs, task).T
            kn = mo @ kN @ mo.T

            dak = kn.T@S@da - da@S@kn.T
            dbk = kn.T@S@db - db@S@kn.T

            (fak, fbk), = self.get_two_el_fock(task, (dak, dbk),)

            kfa = S@kn@fa - fa@kn@S
            kfb = S@kn@fa - fa@kn@S

            fat = fak + kfa
            fbt = fbk + kfb

            gao = S@(da@fat.T + db@fbt.T) - (fat.T@da + fbt.T@db)@S
            gmo = mo.T @ gao @ mo

            gv = - self.mat2vec(gmo, mol_orbs, task)
        else:
            gv = np.zeros(vecs.shape)

            dks = []
            kns = []

            for col in range(vecs.shape[1]):
                vec = vecs[:, col]

                kN = self.vec2mat(vec, mol_orbs, task).T
                kn = mo @ kN @ mo.T

                dak = kn.T@S@da - da@S@kn.T
                dbk = kn.T@S@db - db@S@kn.T

                dks.append((dak, dbk))
                kns.append(kn)

            dks = tuple(dks)
            fks = self.get_two_el_fock(task, *dks)

            for col, (kn, (fak, fbk)) in enumerate(zip(kns, fks)):

                kfa = S@kn@fa - fa@kn@S
                kfb = S@kn@fb - fb@kn@S

                fat = fak + kfa
                fbt = fbk + kfb

                gao = S@(da@fat.T + db@fbt.T) - (fat.T@da + fbt.T@db)@S
                gmo = mo.T @ gao @ mo

                gv[:, col] = -self.mat2vec(gmo, mol_orbs, task)

        return gv


    def s2n(self, vecs, mol_orbs, task):

        b = np.array(vecs)

        S = self.get_overlap(task)
        D = mol_orbs.get_density(task.molecule)
        da = D.alpha_to_numpy(0)
        db = D.beta_to_numpy(0)
        D = da + db
        mo = mol_orbs.alpha_to_numpy()

        if len(b.shape) == 1:
            kappa = self.vec2mat(vecs, mol_orbs, task).T
            kappa_ao = mo @ kappa @ mo.T

            s2n_ao = kappa_ao.T@S@D - D@S@kappa_ao.T
            s2n_mo = mo.T @ S @ s2n_ao @ S@mo
            s2n_vecs = - self.mat2vec(s2n_mo, mol_orbs, task)
        elif len(b.shape) == 2:
            s2n_vecs = np.ndarray(b.shape)
            rows, columns = b.shape
            for c in range(columns):
                kappa = self.vec2mat(b[:, c], mol_orbs, task).T
                kappa_ao = mo @ kappa @ mo.T

                s2n_ao = kappa_ao.T@S@D - D@S@kappa_ao.T
                s2n_mo = mo.T @ S @ s2n_ao @ S@mo
                s2n_vecs[:, c] = - self.mat2vec(s2n_mo, mol_orbs, task)
        return s2n_vecs



