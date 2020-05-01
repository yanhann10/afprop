
class afprop_oop:
    def __init__(
        self,
        mydata,
        num_cluster_pref=1,
        iterations=100,
        damp_c=0.5,
        num_stable_iters=10,
    ):
        self.mydata = mydata
        self.num_cluster_pref = num_cluster_pref
        self.iterations = iterations
        self.damp_c = damp_c
        self.num_stable_iters = num_stable_iters

        
    def fit_predict(self):
        if isinstance(self.mydata, pd.DataFrame):
            self.mydata = self.mydata.values

        # data input error messages
        if self.mydata.shape[1] != 2:
            raise ValueError("S must have two columns.")
        if self.num_cluster_pref != 1 and self.num_cluster_pref != 2:
            raise ValueError(
                "Enter valid indication (1 or 2) of cluster number preference."
            )
        if  self.iterations < 1 or type( self.iterations) != int:
            raise ValueError("Enter a valid number of iterations.")
        if  self.damp_c <= 0 or  self.damp_c > 1:
            raise ValueError("Enter a valid damping constant.")
        if (
            self.num_stable_iters < 1
            or self.num_stable_iters >  self.iterations
            or type (self.num_stable_iters) != int
        ):
            raise ValueError("Enter a valid number of iterations to check for stability.")

        # count number of data points, IE number rows in mydata
        num_data_pts = self.mydata.shape[0]

        s_matrix = calc_similarity_matrix(self.mydata, num_cluster_pref=1)

        # initialize a_array: a(i,k) = 0 at 0th iteration
        a_array = np.zeros(num_data_pts * num_data_pts * (self.iterations)).reshape(
            ( self.iterations, num_data_pts, num_data_pts)
        )

        # initialize r_array
        r_array = np.zeros((self.iterations, num_data_pts, num_data_pts))

        # fill in r_array values for 0th iteration
        r_array[0] = init_r_array(s_matrix)

        ### iterative loop for iterations 1+

        # define tracker variables for checking for stability
        clusters_prev = np.zeros(num_data_pts)
        iter_stability = np.zeros(self.iterations)

        for niter in range(1, self.iterations):

            # update a and r arrays at each iteration
            a_array[niter] = a_array_update(num_data_pts, niter, r_array,  self. damp_c, a_array)
            # r_array[niter] = r_array_update(num_data_pts, niter, a_array, s_matrix, damp_c, r_array)
            r_array[niter] = r_array_update(niter, a_array, s_matrix,  self. damp_c, r_array)
            r_s_sum_array = r_array[niter] + a_array[niter]

            # results of each iteration's clustering attempt
            clusters = np.argmax(
                r_s_sum_array, axis=1
            )  # the list points grouped by their assigned center
            centers = np.where(
                np.argmax(r_s_sum_array, axis=1) == np.array(range(num_data_pts))
            )  # the points that are centers
            is_center = np.argmax(r_s_sum_array, axis=1) == np.array(
                range(num_data_pts)
            )  # true if pt is a center, false otherwise

            # record whether this iteration's clustering is the same as in previous iteration
            if np.array_equal(clusters, clusters_prev):
                iter_stability[niter] = 1

            clusters_prev = clusters

            # if you have seen enough identical clusterings in a row,
            # create a scatterplot illustrating the results
            # and break the iteration loop and return the final clustering results
            if niter > self. num_stable_iters and np.all(
                iter_stability[niter - self. num_stable_iters : niter] == 1
            ):

                centers = np.where(
                    np.argmax(r_s_sum_array, axis=1) == np.array(range(num_data_pts))
                )  # the points that are centers
                is_center = np.argmax(r_s_sum_array, axis=1) == np.array(
                    range(num_data_pts)
                )  # true if pt is a center, false otherwise

                exemplars = centers[0]
                num_clusters = len(np.unique(clusters))
                final_iter = niter + 1

                # if data is 2D, print scatter plot
                if self.mydata.shape[1] == 2:
                    make_cluster_plot(self.mydata, r_s_sum_array, is_center)
                    return clusters, exemplars, num_clusters, final_iter
                else:
                    return clusters, exemplars, num_clusters, final_iter
                break

            # if you have not seen enough identical clusterings in a row by the final iteration,
            # just print a message
            elif niter == self.iterations - 1:
                print("Stability not acheived. Consider reducing self. num_stable_iters.")
