
# # Plot painting
# # --------------------------------------------------------------------------------------

# class Painter:
#     def __init__(
#         self,
#         matrix: np.ndarray,
#         inj_size: int,
#         inj_rms: float = 0.15,
#         inj_cut: float = 3.0,
#         method: str = "correlated",
#     ) -> None:
#         self.M = matrix
        
#         eig_res = np.linalg.eig(M)
#         v1 = normalize_eigvec(eig_res.eigenvectors[:, 0])
#         v2 = normalize_eigvec(eig_res.eigenvectors[:, 2])
#         self.V_inv = build_norm_matrix_from_eigvecs(v1, v2)
#         self.V = np.linalg.inv(self.V_inv)
        
#         self.inj_size = inj_size
#         self.n_turns = n_turns
#         self.inj_rms = inj_rms
#         self.inj_cut = np.repeat(inj_cut, 4)
#         self.t_arr = np.linspace(0.0, 1.0, n_turns + 1)

#         self.method = method
#         self.umax = None  # normalized space
#         self.is_initialized = False

#     def set_umax(self, xmax: np.ndarray) -> None:
#         self.umax = umax

#     def get_u(self, turn: int) -> np.ndarray:
#         t = self.t_arr[turn]
#         if self.method == "correlated":
#             tau = np.sqrt(t)
#             return np.multiply(self.umax, tau)
#         elif self.method == "anti-correlated":
#             tau1 = np.sqrt(1.0 - t)
#             tau2 = np.sqrt(t)
#             return np.multiply(self.umax, [tau1, tau1, tau2, tau2])
#         else:
#             raise ValueError("Invalid method")

#     def gen_pulse(self) -> np.ndarray:
#         return scipy.stats.truncnorm.rvs(
#             scale=self.inj_rms,
#             size=(self.inj_size, 4),
#             a=-self.inj_cut,
#             b=+self.inj_cut,
#         )

#     def paint(self, turns: list[int]) -> np.ndarray:
#         bunches = [self.gen_pulse() for _ in range(turns + 1)]

#         # Place minipulses at origin.
#         for t in range(nturns + 1):
#             bunches[t] += self.get_u(t)

#         # Advance phases.
#         for t, minipulse in enumerate(tqdm(minipulses)):
#             matrix = np.zeros((4, 4))
#             matrix[0:2, 0:2] = rotation_matrix(2.0 * np.pi * self.tune_x * t)
#             matrix[2:4, 2:4] = rotation_matrix(2.0 * np.pi * self.tune_y * t)
#             bunches[t] = np.matmul(bunches[t], matrix.T)

#         # Combine minipulses.
#         bunch = np.vstack(bunches)

#         # Unnormalize coordinates
#         bunch = np.matmul(bunch, self.V.T)

#         return bunch


# argparse.Namespace(
#     turns=2800,
#     stride=200,
#     size=200,
# )

# turns_list = list(range(0, args.nturns + args.stride, args.stride))