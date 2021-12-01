np.random.seed(0)
d = 2
sampled_perturbed = np.load('Project Code/sampled_2D_two_clusters.npy')
data = sampled_perturbed[0 : 1000]
k = 5


#Problems:
#1) Metropolis Hastings term in the truncated normal proposal has to be included? How?
#2) Evaluate wishart gives 0 values"
#3) Update the prior function: integrate out the other values


def sample_uniq_vals_fullcond_Wasserstein(clusdata, lam, uniq_vals, h, key):
    """
    Perform one step of the Metropolis Hastings step to sample a couple (mu,cov) from NIW*Wasserstein
    """
    var_prop = 0.01 #variance for the covariance matrix sampling
    cov_prop = np.eye(d) #covariance to sample the mean

    #Starting point of MH
    mu_old = np.array([uniq_vals[h,0,0], uniq_vals[h,1,0]])
    cov_old = np.array([[uniq_vals[h,0,1], uniq_vals[h,0,2]],
                        [uniq_vals[h,1,1], uniq_vals[h,1,2]]])


    # Compute the parameters of the updated NIW
    mu_n, lam_n, phi_n, nu_n = compute_param_conjugate_NIW(clusdata, lam)

    # Sample the proposal using truncated normal
    mu,cov = sample_proposal(mu_old,cov_old,var_prop,cov_prop )

    # Compute acceptance rate(and return it for tuning)
    beta = compute_beta(uniq_vals, h, mu, cov, mu_n, lam_n, phi_n, nu_n, mu_old, cov_old)
    print("beta")
    print(beta)
    accept_rate = np.minimum(1, beta)
    draw = tfd.Bernoulli(probs = accept_rate).sample()


    # Select the new or old values without an if to help JAX implementation
    if(draw == 0):
        mu = mu_old
        cov = cov_old

    return np.array([[mu[0], cov[0,0], cov[0,1]],[mu[1], cov[1,0], cov[1,1]]]) , accept_rate


def compute_param_conjugate_NIW(clusdata, lam):
    """
    Compute parameters for the conjugate NIW distribution
    """
    n=len(clusdata)
    mean_clus = np.mean(clusdata, axis = 0)
    mean_data = np.mean(data, axis = 0)

    W0 = np.diag(np.ones(d))
    C = (n-1)*np.cov(clusdata, rowvar = False)
    D = lam*n/(lam+n)*np.dot(mean_clus - mean_data, mean_clus - mean_data)
    phi_n = (W0 + C + D)

    nu_n = d + n
    lam_n = lam + n
    mu_n = (lam * mean_data + n * mean_clus)/(lam+n)
    return mu_n, lam_n, phi_n, nu_n

def sample_from_NIW(mu_n, lam_n, phi_n, nu_n):
    """
    Sample from a NIW given its parameters
    """
    W = np.linalg.inv(phi_n)
    chol = np.linalg.cholesky(W)
    prec = tfd.WishartTriL(df=nu_n, scale_tril=chol).sample()
    cov = np.array(tf.linalg.inv(prec))
    mu = tfd.MultivariateNormalFullCovariance(mu_n, cov/(lam_n)).sample()
    return mu, cov

def evaluate_NIW(x_mu, x_cov, mu_n, lam_n, phi_n, nu_n):
    """
    Evaluate the likelihood given the parameters of a NIW
    """
    W = np.linalg.inv(phi_n)
    chol = np.linalg.cholesky(W)
    inv_cov = np.linalg.inv(x_cov)

    f1 = tfd.WishartTriL(df=nu_n, scale_tril=chol).prob(inv_cov)
    f2 = tfd.MultivariateNormalFullCovariance(mu_n, x_cov/(lam_n)).prob(x_mu)
    return f1 * f2

def compute_Wasserstein(mu_1, cov_1, mu_2, cov_2):
    """
    Wasserstein distance for the Gaussian Case
    """
    norm = np.linalg.norm(mu_1 - mu_2, ord = 2)
    sqrt_C2 = scipy.linalg.sqrtm(cov_2)
    C1_sqrt_C2 = np.matmul(cov_1,sqrt_C2)
    sqrt_C2_C1_sqrt_C2 = np.matmul(sqrt_C2,C1_sqrt_C2)
    trace = np.trace(cov_1 + cov_2 - 2 * scipy.linalg.sqrtm(sqrt_C2_C1_sqrt_C2))

    return norm + trace

def sample_proposal(mu_old,cov_old,var_prop,cov_prop):
    """
    Sample from 3 truncated normals to 
    """

    sigma_0 = cov_old[0,0]
    sigma_1 = cov_old[1,1]
    sigma_symm = cov_old[0,1]

    mu = tfd.MultivariateNormalFullCovariance(mu_old , cov_prop).sample()

    p0 = (sigma_symm ** 2)/sigma_1
    sigma_0_new = tfd.TruncatedNormal(loc = sigma_0, scale = var_prop, low = p0,high = 1e6).sample()

    p0_1 = np.sqrt(sigma_0_new * sigma_1)
    sigma_symm_new = tfd.TruncatedNormal(loc = sigma_symm, scale = var_prop, low = -p0_1, high = p0_1).sample()

    p1 = (sigma_symm_new ** 2)/sigma_0_new
    sigma_1_new = tfd.TruncatedNormal(loc = sigma_1, scale = var_prop, low = p1,high = 1e6).sample()

    cov = np.array([[sigma_0_new , sigma_symm_new],[sigma_symm_new , sigma_1_new]])

    return mu,cov


def compute_beta(uniq_vals, h, mu, cov, mu_n, lam_n, phi_n, nu_n, mu_old, cov_old):
    """
    Compute the beta term in the MH algorithm
    """
    # Target distribution terms
    num_1 = evaluate_NIW(mu, cov, mu_n, lam_n, phi_n, nu_n)
    den_1 = evaluate_NIW(mu_old, cov_old , mu_n, lam_n, phi_n, nu_n)

    # Wasserstein Distances
    prod = 1
    for j in range(k):
        if(j != k):
            mu_k = np.array([uniq_vals[j,0,0],uniq_vals[j,1,0]])
            cov_k = np.array( [uniq_vals[j,0,1:3] , uniq_vals[j,1,1:3]]  )
            prod = prod * compute_Wasserstein(mu , cov , mu_k , cov_k) / compute_Wasserstein(mu_n , np.linalg.inv(phi_n) , mu_k, cov_k)



    print("num_1" + str(num_1))
    print("prod" + str(prod))
    print("den_1" + str(den_1))

#     if num_1 < 1e-15 and den_1 < 1e-15:
#         num_1 = 1
#         den_1 = 1
#     elif den_1 < 1e-15:
#         return 0


    return num_1*prod/den_1


def update_cluster_allocs(data, weights, uniq_vals):

    logprobs = tfd.MultivariateNormalFullCovariance(uniq_vals[:,:,0], uniq_vals[:,:,1:3]).log_prob(data[:, np.newaxis])
    logprobs += np.log(weights)
    probs =  np.exp(logprobs)/(np.sum(np.exp(logprobs), axis=1))[:,None]
    for i in range(len(probs)):
        if np.sum(probs[i]) != 1:
            probs[i] = np.zeros(k)
            idx = np.random.randint(0,k)
            probs[i,idx] = 1
    return tfd.Categorical(probs=probs, validate_args=True).sample()


# FUNCTION TO UPDATE THE WEIGHTS
# IT UPDATES THE PARAMETERS OF THE DIRICHLET AND RETURNS THE NEW WEIGHTS
def update_weights(cluster_allocs, n_clus, k, alpha):

    n_by_clus = np.array([np.sum(cluster_allocs == h) for h in range(n_clus)])
    post_params = np.ones(k) * alpha + n_by_clus
    return tfd.Dirichlet(post_params.astype(float)).sample()


def sample_uniq_vals_prior(lam):

    chol = tf.linalg.cholesky(np.diag(np.ones(d)))
    prec = tfd.WishartTriL(df=d, scale_tril=chol).sample()
    var = np.array(tf.linalg.inv(prec))
    mu = tfd.MultivariateNormalFullCovariance(np.mean(np.array(data), axis = 0), var/lam).sample()
    return np.array([[mu[0], var[0,0], var[0,1]],[mu[1], var[1,0], var[1,1]]])


def run_one_gibbs(data, cluster_allocs, uniq_vals, weights, alpha, lam , key):

    """
    Run one gibbs sampler iteration
    Takes in input values of the previous iteration and sample the new values from sample_uniq_vals_fullcond, update_weights and update_cluster_allocs
    Returns:

    -cluster_allocs: for every data point, the cluster assigned
    -uniq_vals: array of parameters of the distributions. Matrix has d rows of the type:[mu[0], var[0,0], var[0,1]]
    -weights: array with the weights of the clusters
     """

    n_clus = len(weights)

    for h in range(n_clus):

        #Extract data assigned to cluster h and sample
        clusdata = data[cluster_allocs == h]
        if len(clusdata) != 0:
            uniq_vals[h, :], acc_rate = sample_uniq_vals_fullcond_Wasserstein(clusdata, lam, uniq_vals, h, key)

    weights = update_weights(cluster_allocs ,n_clus, k, alpha)

    cluster_allocs = update_cluster_allocs(data, weights, uniq_vals)

    return cluster_allocs, uniq_vals, weights, key




def run_mcmc(data, k,  d,key , niter=1000, nburn=300, thin=5 ):

    """
    Runs entire MCMC
    Takes in input data, number of clusters, number of iterations, burn-in and thin
    Returns the parameters recorded after burn-in phase
    """

    b = time.time() # only to measure time


    #Priors
    cluster_allocs = tfd.Categorical(probs=np.ones(k) / k).sample(len(data))
    weights = np.ones(k)/k
    alpha = 0.1
    lam = 0.1
    uniq_vals = np.dstack([
        tfd.MultivariateNormalFullCovariance(np.mean(np.array(data), axis=0), np.linalg.inv(np.diag(np.ones(d))/lam)).sample(k),
        tfd.WishartTriL(df=d, scale_tril=tf.linalg.cholesky(np.diag(np.ones(d))) ).sample(k)])

    #Output values
    allocs_out = []
    uniq_vals_out = []
    weights_out = []

    #Useful value
    data_mean = jnp.mean(data, axis = 0)

    #Iterations
    for i in range(niter):
        cluster_allocs, uniq_vals, weights , key = run_one_gibbs(
            data, cluster_allocs, uniq_vals, weights, alpha, lam , key)

        if i > nburn and i % thin == 0:
            allocs_out.append(cluster_allocs)
            uniq_vals_out.append(uniq_vals.copy())
            weights_out.append(weights)

        if i % 10 == 0:
            a = time.time()
            print("\rIter {0} / {1}".format(i+1, niter) + " Remaining minutes: " + str(round((a-b)*(niter-i)/(60*10) ,1)) , flush=False, end=" ")
            b = time.time()

    return allocs_out, uniq_vals_out, weights_out,key
