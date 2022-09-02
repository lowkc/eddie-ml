# REMOVE COEFFICIENTS FROM LINEAR COMBINATIONS


from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class LocalSimilarityKernel(ABC):
    """An abstract base class for all kernels that use the similarity of local
    atomic environments to comute a global similarity measure.
    """
    def __init__(self, metric, gamma=None, degree=3, coef0=1, kernel_params=None, normalize_kernel=True):
        """
        Args:
            metric(string or callable): The pairwise metric used for
                calculating the local similarity. Accepts any of the sklearn
                pairwise metric strings (e.g. "linear", "rbf", "laplacian",
                "polynomial") or a custom callable. A callable should accept
                two arguments and the keyword arguments passed to this object
                as kernel_params, and should return a floating point number.
            gamma(float): Gamma parameter for the RBF, laplacian, polynomial,
                exponential chi2 and sigmoid kernels. Interpretation of the
                default value is left to the kernel; see the documentation for
                sklearn.metrics.pairwise. Ignored by other kernels.
            degree(float): Degree of the polynomial kernel. Ignored by other
                kernels.
            coef0(float): Zero coefficient for polynomial and sigmoid kernels.
                Ignored by other kernels.
            kernel_params(mapping of string to any): Additional parameters
                (keyword arguments) for kernel function passed as callable
                object.
            normalize_kernel(boolean): Whether to normalize the final global
                similarity kernel. The normalization is achieved by dividing each
                kernel element :math:`K_{ij}` with the factor
                :math:`\sqrt{K_{ii}K_{jj}}`
        """
        self.metric = metric
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.normalize_kernel = normalize_kernel

    def create(self, x, xatoms, y=None, yatoms=None):
        """Creates the kernel matrix based on the given lists of local
        features x and y.
        Args:
            x(iterable): A list of local feature arrays for each structure.
            y(iterable): An optional second list of features. If not specified
                it is assumed that y=x.
        Returns:
            The pairwise global similarity kernel K[i,j] between the given
            structures, in the same order as given in the input, i.e. the
            similarity of structures i and j is given by K[i,j], where features
            for structure i and j were in features[i] and features[j]
            respectively.
        """
        symmetric = False
        if y is None:
            y = x
            yatoms = xatoms
            symmetric = True

        # First calculate the "raw" pairwise similarity of atomic environments
        n_x = len(x)
        n_y = len(y)

        C_ij_AC_dict = {}
        C_ij_AD_dict = {}
        C_ij_BC_dict = {}
        C_ij_BD_dict = {}
        for i in range(n_x):
            for j in range(n_y):

                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                x_i = x[i] # Select dimer i in X
                x_i_atoms = xatoms[i] # Number of atoms in molecule 1 from dimer i
                
                x_i_A = x_i[:x_i_atoms] # Select first molecule
                x_i_B = x_i[x_i_atoms:]

                # Save time on symmetry
                if symmetric and j == i:
                    y_j = None
                    y_j_C = None
                    y_j_D = None
                else:
                    y_i_atoms = yatoms[j]
                    y_j = y[j] # Select dimer in Y
                    y_j_C = y_j[:y_i_atoms]
                    y_j_D = y_j[y_i_atoms:] # Second molecule
                
                # Get pairwise similarity between each atom in each dimer.
                C_ij_AC = self.get_pairwise_matrix(x_i_A, y_j_C)
                C_ij_AD =  self.get_pairwise_matrix(x_i_A, y_j_D)
                C_ij_BC = self.get_pairwise_matrix(x_i_B, y_j_C)
                C_ij_BD = self.get_pairwise_matrix(x_i_B, y_j_D)

                #C_ij_dict[i, j] = C_ij
                C_ij_AC_dict[i, j] = C_ij_AC
                C_ij_AD_dict[i, j] = C_ij_AD
                C_ij_BC_dict[i, j] = C_ij_BC
                C_ij_BD_dict[i, j] = C_ij_BD

        # Calculate the global pairwise similarity between the entire structures
        K_ij_AC = np.zeros((n_x, n_y))
        K_ij_AD = np.zeros((n_x, n_y))
        K_ij_BC = np.zeros((n_x, n_y))
        K_ij_BD = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):

                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                C_ij_AC = C_ij_AC_dict[i, j]
                C_ij_AD = C_ij_AD_dict[i, j]
                C_ij_BC = C_ij_BC_dict[i, j]
                C_ij_BD = C_ij_BD_dict[i, j]

                k_ij_AC = self.get_global_similarity(C_ij_AC)
                k_ij_AD = self.get_global_similarity(C_ij_AD)
                k_ij_BC = self.get_global_similarity(C_ij_BC)
                k_ij_BD = self.get_global_similarity(C_ij_BD)

                K_ij_AC[i, j] = k_ij_AC
                K_ij_AD[i, j] = k_ij_AD
                K_ij_BC[i, j] = k_ij_BC
                K_ij_BD[i, j] = k_ij_BD
                
                
                # Save data also on lower triangular part for symmetric matrices
                if symmetric and j != i:
                    K_ij_AC[j, i] = k_ij_AC
                    K_ij_BD[j, i] = k_ij_BD
                    K_ij_AD[j, i] = k_ij_AD
                    K_ij_BC[j, i] = k_ij_BC

        K_ij = K_ij_AD + K_ij_AC + K_ij_BC + K_ij_BD

        # Enforce kernel normalization if requested.
        if self.normalize_kernel:
            if symmetric:
                k_ii = np.diagonal(K_ij)
                x_k_ii_sqrt = np.sqrt(k_ii)
                y_k_ii_sqrt = x_k_ii_sqrt
            else:
                # Calculate self-similarity for X
                x_k_ii = np.empty(n_x)
                for i in range(n_x):
                    x_i_atoms = xatoms[i]
                    x_i_A = x[i][:x_i_atoms]
                    x_i_B = x[i][x_i_atoms:]
                    C_ii_A = self.get_pairwise_matrix(x_i_A)
                    C_ii_B = self.get_pairwise_matrix(x_i_B)

                    k_ii_A = self.get_global_similarity(C_ii_A)
                    k_ii_B = self.get_global_similarity(C_ii_B)

                    x_k_ii[i] = k_ii_A + k_ii_B

                x_k_ii_sqrt = np.sqrt(x_k_ii)

                # Calculate self-similarity for Y
                y_k_ii = np.empty(n_y)
                for i in range(n_y):
                    y_i_atoms = yatoms[i]
                    y_i_A = y[i][:y_i_atoms]
                    y_i_B = y[i][y_i_atoms:]

                    C_ii_A = self.get_pairwise_matrix(y_i_A)
                    C_ii_B = self.get_pairwise_matrix(y_i_B)

                    k_ii_A = self.get_global_similarity(C_ii_A)
                    k_ii_B = self.get_global_similarity(C_ii_B)

                    y_k_ii[i] = k_ii_A + k_ii_B   
                y_k_ii_sqrt = np.sqrt(y_k_ii)

            K_ij /= np.outer(x_k_ii_sqrt, y_k_ii_sqrt)

        return K_ij

    def get_pairwise_matrix(self, X, Y=None):
        """Calculates the pairwise similarity of atomic environments with
        scikit-learn, and the pairwise metric configured in the constructor.
        Args:
            X(np.ndarray): Feature vector for the atoms in structure A
            Y(np.ndarray): Feature vector for the atoms in structure B
        Returns:
            np.ndarray: NxM matrix of local similarities between structures A
                and B, with N and M atoms respectively.
        """
        if callable(self.metric):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.metric,
                                filter_params=True, **params)


    @abstractmethod
    def get_global_similarity(self, localkernel):
        """
        Computes the global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Global similarity between the structures A and B.
        """


class ElementalSOAPKernel(ABC):
    """An abstract base class for all kernels that use the similarity of local
    atomic environments to comute a global similarity measure.
    """
    def __init__(self, degree=2, alpha=0.5, threshold=1e-6, normalize_kernel=True):
        """
        Used to compute a global similarity of structures based on the
        regularized-entropy match (REMatch) kernel of local atomic environments in
        the structure. More precisely, returns the similarity kernel K as:

    .. math::
        \DeclareMathOperator*{\\argmax}{argmax}
        K(A, B) &= \mathrm{Tr} \mathbf{P}^\\alpha \mathbf{C}(A, B)

        \mathbf{P}^\\alpha &= \\argmax_{\mathbf{P} \in \mathcal{U}(N, N)} \sum_{ij} P_{ij} (1-C_{ij} +\\alpha \ln P_{ij})

        where the similarity between local atomic environments :math:`C_{ij}` has
        been calculated with the pairwise metric (e.g. linear, gaussian) defined by
        the parameters given in the constructor.

        or reference, see:

        "Comparing molecules and solids across structural and alchemical
        space", Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti,
        Phys.  Chem. Chem. Phys. 18, 13754 (2016),
        https://doi.org/10.1039/c6cp00415f
        Args:
            alpha(float): Parameter controlling the entropic penalty. Values
                close to zero approach the best-match solution and values
                towards infinity approach the average kernel.
            threshold(float): Convergence threshold used in the
                Sinkhorn-algorithm.
            normalize_kernel(boolean): Whether to normalize the final global
                similarity kernel. The normalization is achieved by dividing each
                kernel element :math:`K_{ij}` with the factor
                :math:`\sqrt{K_{ii}K_{jj}}`
        """
        self.degree = degree
        self.normalize_kernel = normalize_kernel
        self.alpha = alpha
        self.threshold = threshold

    def create(self, x, xatoms, xatomlist, y=None, yatoms=None, yatomlist=None):
        """Creates the kernel matrix based on the given lists of local
        features x and y.
        Args:
            x(iterable): A list of local feature arrays for each structure.
            y(iterable): An optional second list of features. If not specified
                it is assumed that y=x.
        Returns:
            The pairwise global similarity kernel K[i,j] between the given
            structures, in the same order as given in the input, i.e. the
            similarity of structures i and j is given by K[i,j], where features
            for structure i and j were in features[i] and features[j]
            respectively.
        """
        symmetric = False
        if y is None:
            y = x
            yatoms = xatoms
            yatomlist = xatomlist
            symmetric = True

        # First calculate the "raw" pairwise similarity of atomic environments
        n_x = len(x)
        n_y = len(y)

        C_ij_AC_dict = {}
        C_ij_AD_dict = {}
        C_ij_BC_dict = {}
        C_ij_BD_dict = {}
        for i in range(n_x):
            for j in range(n_y):
                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                x_i = x[i] # Select dimer i in X
                x_i_atoms = xatoms[i] # Number of atoms in molecule 1 from dimer i
                x_i_atomlist = xatomlist[i] # Elements in molecule 1 from dimer i
                
                x_i_A = x_i[:x_i_atoms] # Select first molecule descriptor
                x_i_B = x_i[x_i_atoms:] # Second molecule descriptor

                x_i_Aatoms = x_i_atomlist[:x_i_atoms] # Elements in first molecule
                x_i_Batoms = x_i_atomlist[x_i_atoms:] # Elements in second molecule

                # x_i_A = x_i_A/len(x_i_Aatoms)
                # x_i_B = x_i_B/len(x_i_Batoms)

                # Save time on symmetry
                if symmetric and j == i:
                    y_j = None
                    y_j_C = None
                    y_j_D = None
                    y_j_Catoms = None
                    y_j_Datoms = None
                else:
                    y_j = y[j] # Select dimer in Y
                    y_j_atoms = yatoms[j]
                    y_j_atomlist = yatomlist[j]
                    y_j_C = y_j[:y_j_atoms]
                    y_j_D = y_j[y_j_atoms:] 
                    y_j_Catoms = y_j_atomlist[:y_j_atoms] # Elements in first molecule
                    y_j_Datoms = y_j_atomlist[y_j_atoms:] # Elements in second molecule

                    # y_j_C = y_j_C/len(y_j_Catoms)
                    # y_j_D = y_j_D/len(y_j_Datoms)
                
                # Get pairwise similarity between each atom in each dimer.
                C_ij_AC = self.get_pairwise_matrix(x_i_A, x_i_Aatoms, y_j_C, y_j_Catoms)
                C_ij_AD =  self.get_pairwise_matrix(x_i_A,x_i_Aatoms, y_j_D, y_j_Datoms)
                C_ij_BC = self.get_pairwise_matrix(x_i_B, x_i_Batoms, y_j_C, y_j_Catoms)
                C_ij_BD = self.get_pairwise_matrix(x_i_B, x_i_Batoms, y_j_D, y_j_Datoms)

                #C_ij_dict[i, j] = C_ij
                C_ij_AC_dict[i, j] = C_ij_AC
                C_ij_AD_dict[i, j] = C_ij_AD
                C_ij_BC_dict[i, j] = C_ij_BC
                C_ij_BD_dict[i, j] = C_ij_BD

        # Calculate the global pairwise similarity between the entire structures
        K_ij_AC = np.zeros((n_x, n_y))
        K_ij_AD = np.zeros((n_x, n_y))
        K_ij_BC = np.zeros((n_x, n_y))
        K_ij_BD = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):

                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                C_ij_AC = C_ij_AC_dict[i, j]
                C_ij_AD = C_ij_AD_dict[i, j]
                C_ij_BC = C_ij_BC_dict[i, j]
                C_ij_BD = C_ij_BD_dict[i, j]

                k_ij_AC = self.get_global_similarity(C_ij_AC)
                k_ij_AD = self.get_global_similarity(C_ij_AD)
                k_ij_BC = self.get_global_similarity(C_ij_BC)
                k_ij_BD = self.get_global_similarity(C_ij_BD)

                #K_ij[i, j] = k_ij
                K_ij_AC[i, j] = k_ij_AC
                K_ij_AD[i, j] = k_ij_AD
                K_ij_BC[i, j] = k_ij_BC
                K_ij_BD[i, j] = k_ij_BD
                
                # Save data also on lower triangular part for symmetric matrices
                if symmetric and j != i:
                    #K_ij[j, i] = k_ij
                    K_ij_AC[j, i] = k_ij_AC
                    K_ij_BD[j, i] = k_ij_BD
                    K_ij_AD[j, i] = k_ij_AD
                    K_ij_BC[j, i] = k_ij_BC

        K_ij = K_ij_AD + K_ij_AC + K_ij_BC + K_ij_BD

        # Enforce kernel normalization if requested.
        if self.normalize_kernel:
            if symmetric:
                k_ii = np.diagonal(K_ij)
                x_k_ii_sqrt = np.sqrt(k_ii)
                y_k_ii_sqrt = x_k_ii_sqrt
            else:
                # Calculate self-similarity for X
                x_k_ii = np.empty(n_x)
                for i in range(n_x):
                    x_i_atoms = xatoms[i]
                    x_i_A = x[i][:x_i_atoms]
                    x_i_B = x[i][x_i_atoms:]
                    x_i_atomlist = xatomlist[i]
                    x_i_Aatoms = x_i_atomlist[:x_i_atoms] # Elements in first molecule
                    x_i_Batoms = x_i_atomlist[x_i_atoms:] # Elements in second molecule

                    C_ii_A = self.get_pairwise_matrix(x_i_A, x_i_Aatoms)
                    C_ii_B = self.get_pairwise_matrix(x_i_B, x_i_Batoms)

                    k_ii_A = self.get_global_similarity(C_ii_A)
                    k_ii_B = self.get_global_similarity(C_ii_B)

                    x_k_ii[i] = k_ii_A + k_ii_B

                x_k_ii_sqrt = np.sqrt(x_k_ii)

                # Calculate self-similarity for Y
                y_k_ii = np.empty(n_y)
                for i in range(n_y):
                    y_i_atoms = yatoms[i]
                    y_i_C = y[i][:y_i_atoms]
                    y_i_D = y[i][y_i_atoms:]
                    y_i_atomlist = yatomlist[i]
                    y_i_Catoms = y_i_atomlist[:y_i_atoms] # Elements in first molecule
                    y_i_Datoms = y_i_atomlist[y_i_atoms:] # Elements in second molecule

                    C_ii_C = self.get_pairwise_matrix(y_i_C, y_i_Catoms)
                    C_ii_D = self.get_pairwise_matrix(y_i_D, y_i_Datoms)

                    k_ii_C = self.get_global_similarity(C_ii_C)
                    k_ii_D = self.get_global_similarity(C_ii_D)

                    y_k_ii[i] = k_ii_C + k_ii_D   
                y_k_ii_sqrt = np.sqrt(y_k_ii)

            K_ij /= np.outer(x_k_ii_sqrt, y_k_ii_sqrt)

        return K_ij

    def get_atomic_filter(self, atomlist1, atomlist2):
        matrix = np.zeros((len(atomlist1), len(atomlist2)))
        for i in range(len(atomlist1)):
            for j in range(len(atomlist2)):
                if atomlist1[i] == atomlist2[j]:
                    #number = data.atomic_numbers[atomlist1[i]]
                    #matrix[i, j] = data.covalent_radii[number]
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0
        return matrix
    
    def get_pairwise_matrix(self, X, Xatomlist, Y=None, Yatomlist=None):
        """Calculates the pairwise similarity of atomic environments with
        scikit-learn, and the pairwise metric configured in the constructor.
        Args:
            X(np.ndarray): Feature vector for the atoms in structure A
            Y(np.ndarray): Feature vector for the atoms in structure B
        Returns:
            np.ndarray: NxM matrix of local similarities between structures A
                and B, with N and M atoms respectively.
        """
        if Yatomlist is None:
            Yatomlist = Xatomlist
        if Y is None:
            Y = X

        kernel = np.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)
        for i, A in enumerate(X):
            for j, B in enumerate(Y):
                kernel[i, j] = local_kernel(A, B, self.degree)
        
        filter_matrix = self.get_atomic_filter(Xatomlist, Yatomlist)
        filtered_kernel = np.multiply(kernel, filter_matrix)
        return filtered_kernel

    def get_global_similarity(self, localkernel):
        """
        Computes the global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Global similarity between the structures A and B.
        """
        n, m = localkernel.shape
        K = np.exp(-(1 - localkernel) / self.alpha)

        # initialisation
        u = np.ones((n,)) / n
        v = np.ones((m,)) / m

        en = np.ones((n,)) / float(n)
        em = np.ones((m,)) / float(m)

        # converge balancing vectors u and v
        itercount = 0
        error = 1
        while (error > self.threshold):
            uprev = u
            vprev = v
            v = np.divide(em, np.dot(K.T, u))
            u = np.divide(en, np.dot(K, v))

            # determine error every now and then
            if itercount % 5:
                error = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            itercount += 1

        # using Tr(X.T Y) = Sum[ij](Xij * Yij)
        # P.T * C
        # P_ij = u_i * v_j * K_ij
        pity = np.multiply( np.multiply(K, u.reshape((-1, 1))), v)

        glosim = np.sum( np.multiply( pity, localkernel))
        return glosim
        #K_ij = np.sum(localkernel)

        #return K_ij 


def local_kernel(X, Y, degree):
    p1 = np.array(X) 
    p2 = np.array(Y)
    p12 = np.dot(p1,p2)**degree
    p11 = np.dot(p1,p1)**degree
    p22 = np.dot(p2,p2)**degree
    return p12/(np.sqrt(p11*p22)+1e-20)


class ElementalSimilarityKernel(ABC):
    """An abstract base class for all kernels that use the similarity of local
    atomic environments to comute a global similarity measure.
    """
    def __init__(self, metric, gamma=None, degree=3, coef0=1, kernel_params=None, normalize_kernel=True):
        """
        Args:
            metric(string or callable): The pairwise metric used for
                calculating the local similarity. Accepts any of the sklearn
                pairwise metric strings (e.g. "linear", "rbf", "laplacian",
                "polynomial") or a custom callable. A callable should accept
                two arguments and the keyword arguments passed to this object
                as kernel_params, and should return a floating point number.
            gamma(float): Gamma parameter for the RBF, laplacian, polynomial,
                exponential chi2 and sigmoid kernels. Interpretation of the
                default value is left to the kernel; see the documentation for
                sklearn.metrics.pairwise. Ignored by other kernels.
            degree(float): Degree of the polynomial kernel. Ignored by other
                kernels.
            coef0(float): Zero coefficient for polynomial and sigmoid kernels.
                Ignored by other kernels.
            kernel_params(mapping of string to any): Additional parameters
                (keyword arguments) for kernel function passed as callable
                object.
            normalize_kernel(boolean): Whether to normalize the final global
                similarity kernel. The normalization is achieved by dividing each
                kernel element :math:`K_{ij}` with the factor
                :math:`\sqrt{K_{ii}K_{jj}}`
        """
        self.metric = metric
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.normalize_kernel = normalize_kernel

    def create(self, x, xatoms, xatomlist, y=None, yatoms=None, yatomlist=None):
        """Creates the kernel matrix based on the given lists of local
        features x and y.
        Args:
            x(iterable): A list of local feature arrays for each structure.
            y(iterable): An optional second list of features. If not specified
                it is assumed that y=x.
        Returns:
            The pairwise global similarity kernel K[i,j] between the given
            structures, in the same order as given in the input, i.e. the
            similarity of structures i and j is given by K[i,j], where features
            for structure i and j were in features[i] and features[j]
            respectively.
        """
        symmetric = False
        if y is None:
            y = x
            yatoms = xatoms
            yatomlist = xatomlist
            symmetric = True

        # First calculate the "raw" pairwise similarity of atomic environments
        n_x = len(x)
        n_y = len(y)

        C_ij_AC_dict = {}
        C_ij_AD_dict = {}
        C_ij_BC_dict = {}
        C_ij_BD_dict = {}
        for i in range(n_x):
            for j in range(n_y):
                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                x_i = x[i] # Select dimer i in X
                x_i_atoms = xatoms[i] # Number of atoms in molecule 1 from dimer i
                x_i_atomlist = xatomlist[i] # Elements in molecule 1 from dimer i
                
                x_i_A = x_i[:x_i_atoms] # Select first molecule descriptor
                x_i_B = x_i[x_i_atoms:] # Second molecule descriptor

                x_i_Aatoms = x_i_atomlist[:x_i_atoms] # Elements in first molecule
                x_i_Batoms = x_i_atomlist[x_i_atoms:] # Elements in second molecule

                # Save time on symmetry
                if symmetric and j == i:
                    y_j = None
                    y_j_C = None
                    y_j_D = None
                    y_j_Catoms = None
                    y_j_Datoms = None
                else:
                    y_j = y[j] # Select dimer in Y
                    y_j_atoms = yatoms[j]
                    y_j_atomlist = yatomlist[j]
                    y_j_C = y_j[:y_j_atoms]
                    y_j_D = y_j[y_j_atoms:] 
                    y_j_Catoms = y_j_atomlist[:y_j_atoms] # Elements in first molecule
                    y_j_Datoms = y_j_atomlist[y_j_atoms:] # Elements in second molecule

                    # y_j_C = y_j_C/len(y_j_Catoms)
                    # y_j_D = y_j_D/len(y_j_Datoms)
                
                # Get pairwise similarity between each atom in each dimer.
                C_ij_AC = self.get_pairwise_matrix(x_i_A, x_i_Aatoms, y_j_C, y_j_Catoms)
                C_ij_AD =  self.get_pairwise_matrix(x_i_A,x_i_Aatoms, y_j_D, y_j_Datoms)
                C_ij_BC = self.get_pairwise_matrix(x_i_B, x_i_Batoms, y_j_C, y_j_Catoms)
                C_ij_BD = self.get_pairwise_matrix(x_i_B, x_i_Batoms, y_j_D, y_j_Datoms)

                #C_ij_dict[i, j] = C_ij
                C_ij_AC_dict[i, j] = C_ij_AC
                C_ij_AD_dict[i, j] = C_ij_AD
                C_ij_BC_dict[i, j] = C_ij_BC
                C_ij_BD_dict[i, j] = C_ij_BD

        # Calculate the global pairwise similarity between the entire structures
        K_ij_AC = np.zeros((n_x, n_y))
        K_ij_AD = np.zeros((n_x, n_y))
        K_ij_BC = np.zeros((n_x, n_y))
        K_ij_BD = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):

                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                C_ij_AC = C_ij_AC_dict[i, j]
                C_ij_AD = C_ij_AD_dict[i, j]
                C_ij_BC = C_ij_BC_dict[i, j]
                C_ij_BD = C_ij_BD_dict[i, j]

                k_ij_AC = self.get_global_similarity(C_ij_AC)
                k_ij_AD = self.get_global_similarity(C_ij_AD)
                k_ij_BC = self.get_global_similarity(C_ij_BC)
                k_ij_BD = self.get_global_similarity(C_ij_BD)

                #K_ij[i, j] = k_ij
                K_ij_AC[i, j] = k_ij_AC
                K_ij_AD[i, j] = k_ij_AD
                K_ij_BC[i, j] = k_ij_BC
                K_ij_BD[i, j] = k_ij_BD
                
                # Save data also on lower triangular part for symmetric matrices
                if symmetric and j != i:
                    #K_ij[j, i] = k_ij
                    K_ij_AC[j, i] = k_ij_AC
                    K_ij_BD[j, i] = k_ij_BD
                    K_ij_AD[j, i] = k_ij_AD
                    K_ij_BC[j, i] = k_ij_BC

        K_ij = K_ij_AD + K_ij_AC + K_ij_BC + K_ij_BD

        # Enforce kernel normalization if requested.
        if self.normalize_kernel:
            if symmetric:
                k_ii = np.diagonal(K_ij)
                x_k_ii_sqrt = np.sqrt(k_ii)
                y_k_ii_sqrt = x_k_ii_sqrt
            else:
                # Calculate self-similarity for X
                x_k_ii = np.empty(n_x)
                for i in range(n_x):
                    x_i_atoms = xatoms[i]
                    x_i_A = x[i][:x_i_atoms]
                    x_i_B = x[i][x_i_atoms:]
                    x_i_atomlist = xatomlist[i]
                    x_i_Aatoms = x_i_atomlist[:x_i_atoms] # Elements in first molecule
                    x_i_Batoms = x_i_atomlist[x_i_atoms:] # Elements in second molecule

                    C_ii_A = self.get_pairwise_matrix(x_i_A, x_i_Aatoms)
                    C_ii_B = self.get_pairwise_matrix(x_i_B, x_i_Batoms)

                    k_ii_A = self.get_global_similarity(C_ii_A)
                    k_ii_B = self.get_global_similarity(C_ii_B)

                    x_k_ii[i] = k_ii_A + k_ii_B

                x_k_ii_sqrt = np.sqrt(x_k_ii)

                # Calculate self-similarity for Y
                y_k_ii = np.empty(n_y)
                for i in range(n_y):
                    y_i_atoms = yatoms[i]
                    y_i_C = y[i][:y_i_atoms]
                    y_i_D = y[i][y_i_atoms:]
                    y_i_atomlist = yatomlist[i]
                    y_i_Catoms = y_i_atomlist[:y_i_atoms] # Elements in first molecule
                    y_i_Datoms = y_i_atomlist[y_i_atoms:] # Elements in second molecule

                    C_ii_C = self.get_pairwise_matrix(y_i_C, y_i_Catoms)
                    C_ii_D = self.get_pairwise_matrix(y_i_D, y_i_Datoms)

                    k_ii_C = self.get_global_similarity(C_ii_C)
                    k_ii_D = self.get_global_similarity(C_ii_D)

                    y_k_ii[i] = k_ii_C + k_ii_D   
                y_k_ii_sqrt = np.sqrt(y_k_ii)

            K_ij /= np.outer(x_k_ii_sqrt, y_k_ii_sqrt)
        return K_ij

    def get_atomic_filter(self, atomlist1, atomlist2):
        matrix = np.zeros((len(atomlist1), len(atomlist2)))
        for i in range(len(atomlist1)):
            for j in range(len(atomlist2)):
                if atomlist1[i] == atomlist2[j]:
                    #number = data.atomic_numbers[atomlist1[i]]
                    #matrix[i, j] = data.covalent_radii[number]
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0
        return matrix
    
    def get_pairwise_matrix(self, X, Xatomlist, Y=None, Yatomlist=None):
        """Calculates the pairwise similarity of atomic environments with
        scikit-learn, and the pairwise metric configured in the constructor.
        Args:
            X(np.ndarray): Feature vector for the atoms in structure A
            Y(np.ndarray): Feature vector for the atoms in structure B
        Returns:
            np.ndarray: NxM matrix of local similarities between structures A
                and B, with N and M atoms respectively.
        """
        if Yatomlist is None:
            Yatomlist = Xatomlist
        if Y is None:
            Y = X
        filter_matrix = self.get_atomic_filter(Xatomlist, Yatomlist)
        if callable(self.metric):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        kernel = pairwise_kernels(X, Y, metric=self.metric,filter_params=True, **params)
        filtered_kernel = np.multiply(kernel, filter_matrix)
        return filtered_kernel


    def get_global_similarity(self, localkernel):
        """
        Computes the global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Global similarity between the structures A and B.
        """
        K_ij = np.sum(localkernel)

        return K_ij     


class IntermolecularSumKernel(LocalSimilarityKernel):
    """
    Used to compute a global similarity of structures based on the average
    similarity of local atomic environments in the structure. More precisely,
    returns the similarity kernel K as:
    .. math::
        K(A, B) = \\frac{1}{N M}\sum_{ij} C_{ij}(A, B)
    where :math:`N` is the number of atoms in structure :math:`A`, :math:`M` is
    the number of atoms in structure :math:`B` and the similarity between local
    atomic environments :math:`C_{ij}` has been calculated with the pairwise
    metric (e.g. linear, gaussian) defined by the parameters given in the
    constructor.
    """
    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Average similarity between the structures A and B.
        """
        K_ij = np.sum(localkernel)

        return K_ij       

class ElementalDimerSumKernel(ElementalSimilarityKernel):
    """
    Used to compute a global similarity of structures based on the average
    similarity of local atomic environments in the structure. More precisely,
    returns the similarity kernel K as:
    .. math::
        K(A, B) = \\frac{1}{N M}\sum_{ij} C_{ij}(A, B)
    where :math:`N` is the number of atoms in structure :math:`A`, :math:`M` is
    the number of atoms in structure :math:`B` and the similarity between local
    atomic environments :math:`C_{ij}` has been calculated with the pairwise
    metric (e.g. linear, gaussian) defined by the parameters given in the
    constructor.
    """
    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Average similarity between the structures A and B.
        """
        K_ij = np.sum(localkernel)

        return K_ij       


class IntermolecularAverageKernel(LocalSimilarityKernel):
    """
    Used to compute a global similarity of structures based on the average
    similarity of local atomic environments in the structure. More precisely,
    returns the similarity kernel K as:
    .. math::
        K(A, B) = \\frac{1}{N M}\sum_{ij} C_{ij}(A, B)
    where :math:`N` is the number of atoms in structure :math:`A`, :math:`M` is
    the number of atoms in structure :math:`B` and the similarity between local
    atomic environments :math:`C_{ij}` has been calculated with the pairwise
    metric (e.g. linear, gaussian) defined by the parameters given in the
    constructor.
    """
    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Average similarity between the structures A and B.
        """
        K_ij = np.mean(localkernel)

        return K_ij

class ElementalDimerAverageKernel(ElementalSimilarityKernel):
    """
    Used to compute a global similarity of structures based on the average
    similarity of local atomic environments in the structure. More precisely,
    returns the similarity kernel K as:
    .. math::
        K(A, B) = \\frac{1}{N M}\sum_{ij} C_{ij}(A, B)
    where :math:`N` is the number of atoms in structure :math:`A`, :math:`M` is
    the number of atoms in structure :math:`B` and the similarity between local
    atomic environments :math:`C_{ij}` has been calculated with the pairwise
    metric (e.g. linear, gaussian) defined by the parameters given in the
    constructor.
    """
    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Average similarity between the structures A and B.
        """
        K_ij = np.mean(localkernel)

        return K_ij


class IntermolecularRematchKernel(LocalSimilarityKernel):
    def __init__(self, alpha=0.1, threshold=1e-6, metric="linear", gamma=None, degree=3, coef0=1, kernel_params=None, normalize_kernel=True):
        """
        Args:
            alpha(float): Parameter controlling the entropic penalty. Values
                close to zero approach the best-match solution and values
                towards infinity approach the average kernel.
            threshold(float): Convergence threshold used in the
                Sinkhorn-algorithm.
            metric(string or callable): The pairwise metric used for
                calculating the local similarity. Accepts any of the sklearn
                pairwise metric strings (e.g. "linear", "rbf", "laplacian",
                "polynomial") or a custom callable. A callable should accept
                two arguments and the keyword arguments passed to this object
                as kernel_params, and should return a floating point number.
            gamma(float): Gamma parameter for the RBF, laplacian, polynomial,
                exponential chi2 and sigmoid kernels. Interpretation of the
                default value is left to the kernel; see the documentation for
                sklearn.metrics.pairwise. Ignored by other kernels.
            degree(float): Degree of the polynomial kernel. Ignored by other
                kernels.
            coef0(float): Zero coefficient for polynomial and sigmoid kernels.
                Ignored by other kernels.
            kernel_params(mapping of string to any): Additional parameters
                (keyword arguments) for kernel function passed as callable
                object.
            normalize_kernel(boolean): Whether to normalize the final global
                similarity kernel. The normalization is achieved by dividing each
                kernel element :math:`K_{ij}` with the factor
                :math:`\sqrt{K_{ii}K_{jj}}`
        """
        self.alpha = alpha
        self.threshold = threshold
        super().__init__(metric, gamma, degree, coef0, kernel_params, normalize_kernel)

    def get_global_similarity(self, localkernel):
        """
        Computes the REMatch similarity between two structures A and B.
        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: REMatch similarity between the structures A and B.
        """
        n, m = localkernel.shape
        K = np.exp(-(1 - localkernel) / self.alpha)

        # initialisation
        u = np.ones((n,)) / n
        v = np.ones((m,)) / m

        en = np.ones((n,)) / float(n)
        em = np.ones((m,)) / float(m)

        # converge balancing vectors u and v
        itercount = 0
        error = 1
        while (error > self.threshold):
            uprev = u
            vprev = v
            v = np.divide(em, np.dot(K.T, u))
            u = np.divide(en, np.dot(K, v))

            # determine error every now and then
            if itercount % 5:
                error = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            itercount += 1

        # using Tr(X.T Y) = Sum[ij](Xij * Yij)
        # P.T * C
        # P_ij = u_i * v_j * K_ij
        pity = np.multiply( np.multiply(K, u.reshape((-1, 1))), v)

        glosim = np.sum( np.multiply( pity, localkernel))

        return glosim
