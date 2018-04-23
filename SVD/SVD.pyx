import numpy as np
cimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
cimport cython

@cython.cdivision(True)    # turn off nan check in division
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for function
def SVD_SGD(df_data, 
  int n_factors=1, int maxUserId=-1, int maxItemId=-1,
  double alpha=0.01, int n_epochs=40, double e_tol=0.0, double epsilon=0.1, 
  double lam_p=0.0, double lam_t=0.0, double lam_u=0.0, double lam_i=0.0,
  bu_init=None, bi_init=None, p_init=None, t_init=None):
  '''
  SGD

  Performs Stochastic Gradient Descent to find an approximate solution
  to the SVD problem. Does not find true SVD because regularization used.

  PARAMETER:
    df_data: pandas dataframe with 'userId', 'movieId', and 'rating' 
        columns
    n_factors: positive integer for number of factors to keep
    maxUserId: positive integer for maximum user ID in set
    maxItemId: positive integer for maximum item ID in set
    alpha: positive double for learning rate
    n_epochs: positive integer for maximum loops through ratings set
    e_tol: positive double for error tolerance (stops early if 
      two consecutive epochs differ by less than e_tol)
    epsilon: small positive double setting the initialized randomness
    lam_p: positive double for p regularization
    lam_t: positive double for t regularization
    lam_u: positive double for bu regularization
    lam_i: positive double for bi regularization
    bu_init: numpy array (maxUserId) of bu initialization values
    bi_init: numpy array (maxItemId) of bi initialization values
    p_init: numpy array (maxUserId, n_factors) of p initialization values
    t_init: numpy array (maxItemId, n_factors) of t initialization values

  OUTPUT:
    e_arr: numpy array (1D) of MAE for each epoch
    mu: double of mean ratting
    bu_arr: numpy array (maxUserId) of best user baseline
    bi_arr: numpy array (maxItemId) of best item baseline
    p_arr: numpy array (maxUserId, n_factors) of best user preferences p
    t_arr: numpy array (maxItemId, n_factors) of best item topics t

  '''
    
  # Declarations for ints and doubles
  cdef int epoch, idx_rating, n_ratings             # loop vars
  cdef int userId, movieId, u, i, factor            # id vars
  cdef int n_users, n_items                         # N u's and i's
  cdef double n_Iu, n_Ui                            # N i's rated by u
                                                    # N u's rating i
  cdef double r_ui, r_pred_ui, e_ui, e_avg          # rating, pred, err
  cdef double mu                                    # mean rating
  cdef double dbu, dbi                              # baseline update
  cdef double p_u_dot_t_i                           # p_u*t_i
  cdef double e_prev, de                            # MAE prev/ d(MAE)

  # Declarations of int and double ndarrays
  cdef np.ndarray[np.int_t, ndim=1] users, items    # user/item arrs
  cdef np.ndarray[np.int_t, ndim=1] items_to_i, users_to_u
                                                    # conversion arrs
  cdef np.ndarray[np.int_t,ndim=1] num_U, num_I     # n Users/Items
  cdef np.ndarray[np.double_t, ndim=2] p, t         # prefs/topics
  cdef np.ndarray[np.double_t, ndim=1] bu, bi       # user/item base
  cdef np.ndarray[np.double_t, ndim=1] dp, dt       # pref/topic update
  cdef np.ndarray[np.int_t, ndim=1] userIds, itemIds# user/item id arr
  cdef np.ndarray[np.double_t, ndim=1] ratings      # rating arr

  # Initializes arrays of users and items as well as numbers for
  # number of ratings, users, and items
  users = np.sort(df_data.userId.unique())
  items = np.sort(df_data.movieId.unique())
  n_ratings = len(df_data)
  n_users = len(users)
  n_items = len(items)

  # Randomly initialize the user baseline, item baseline, user 
  # preferences matrix, and item topic matrix
  mu = np.mean(df_data['rating'])
  if bu_init is None:
    bu = np.random.normal(0, epsilon, n_users)
  else:
    bu = bu_init[users]
  if bi_init is None:
    bi = np.random.normal(0, epsilon, n_items)
  else:
    bi = bi_init[items]
  if p_init is None:
    p = np.random.normal(0, epsilon, (n_users, n_factors))
  else:
    p = p_init[users]
  if t_init is None:
    t = np.random.normal(0, epsilon, (n_items, n_factors))
  else:
    t = t_init[items]

  # Initializes array to convert between userId and uth row in p
  # and itemId and ith row of t. Must be done because non-consecutive
  # userId and itemId's.
  if maxItemId < np.max(items):
    maxItemId = np.max(items)
  if maxUserId < np.max(users):
    maxUserId = np.max(users)
  items_to_i = np.zeros(maxItemId+1,dtype=int)
  items_to_i[items] = np.arange(n_items)
  users_to_u = np.zeros(maxUserId+1,dtype=int)
  users_to_u[users] = np.arange(n_users)
 
  # Initializes array such that:
  # num_U[i] = number of users rating item i
  # num_I[u] = number of items rated by user u
  num_U = df_data.groupby('movieId').count()['userId'].values
  num_I = df_data.groupby('userId').count()['movieId'].values
  
  # Intializes c++ arrays for storing dataframe
  userIds = np.zeros(n_ratings, dtype=np.int)
  movieIds = np.zeros(n_ratings, dtype=np.int)
  ratings = np.zeros(n_ratings, dtype=np.double)

  # Initializes update vectors for preferences and topics
  dp = np.zeros(n_factors, dtype=np.double)
  dt = np.zeros(n_factors, dtype=np.double)

  # Initializes MAE array
  e_arr = []

  # Optimization procedure
  epoch = 0
  e_prev = 0               # previous MAE
  de = e_tol + 1           # difference in MAE
  while epoch < n_epochs and de > e_tol:

    # randomize the order of df_data
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    userIds = df_data['userId'].values
    movieIds = df_data['movieId'].values
    ratings = df_data['rating'].values
    
    e_avg = 0.0
    for idx_rating in range(n_ratings):
            
      # Finds rows of p and t corresponding to userId and movieId (u and i)
      userId = userIds[idx_rating]
      movieId = movieIds[idx_rating]
      r_ui = ratings[idx_rating]
      u = users_to_u[userId]
      i = items_to_i[movieId]
      n_Iu = num_I[u]
      n_Ui = num_U[i]
     
      # Calculates error
      p_u_dot_t_i = 0.0
      for factor in range(n_factors):
        p_u_dot_t_i += p[u,factor]*t[i,factor]
      r_pred_ui = mu+bu[u]+bi[i]+p_u_dot_t_i
      e_ui = r_ui - r_pred_ui

      # Differential latent vectors p_u and t_i and bias terms
      dbu = alpha*(e_ui-lam_u*bu[u]/n_Iu)
      dbi = alpha*(e_ui-lam_i*bi[i]/n_Ui)
      for factor in range(n_factors):
        dp[factor] = alpha*(e_ui*t[i,factor]-lam_p*p[u,factor]/n_Iu)
        dt[factor] = alpha*(e_ui*p[u,factor]-lam_t*t[i,factor]/n_Ui)
     
      # Updates latent vectors and bias terms
      bu[u] += dbu
      bi[i] += dbi
      for factor in range(n_factors):
        p[u,factor] += dp[factor]
        t[i,factor] += dt[factor]

      # sums absolute errors
      e_avg += abs(e_ui)

    # Finds average error for epoch
    e_avg = e_avg/float(n_ratings)
    e_arr.append(e_avg)

    # Increments de and epoch
    if epoch > 3:
      de = abs(e_avg - e_prev)
    e_prev = e_avg
    epoch += 1

  e_arr = np.array(e_arr)
  bu_arr = np.zeros(maxUserId+1)
  bu_arr[users] = bu
  bi_arr = np.zeros(maxItemId+1)
  bi_arr[items] = bi
  p_arr = np.zeros((maxUserId+1, n_factors))
  p_arr[users] = p
  t_arr = np.zeros((maxItemId+1, n_factors))
  t_arr[items] = t

  
  return (e_arr, mu, bu_arr, bi_arr, p_arr, t_arr)



def SVD_SGD_python(df_data,
  n_factors=1, maxUserId=-1, maxItemId=-1,
  alpha=0.01, n_epochs=40, e_tol=0.0, epsilon=0.1,
  lam_p=0.0, lam_t=0.0, lam_u=0.0, lam_i=0.0,
  bu_init=None, bi_init=None, p_init=None, t_init=None):
  '''
  SGD

  Performs Stochastic Gradient Descent to find an approximate solution
  to the SVD problem. Does not find true SVD because regularization used.

  PARAMETER:
    df_data: pandas dataframe with 'userId', 'movieId', and 'rating' 
        columns
    n_factors: positive integer for number of factors to keep
    maxUserId: positive integer for maximum user ID in set
    maxItemId: positive integer for maximum item ID in set
    alpha: positive double for learning rate
    n_epochs: positive integer for maximum loops through ratings set
    e_tol: positive double for error tolerance (stops early if 
      two consecutive epochs differ by less than e_tol)
    epsilon: small positive double setting the initialized randomness
    lam_p: positive double for p regularization
    lam_t: positive double for t regularization
    lam_u: positive double for bu regularization
    lam_i: positive double for bi regularization
    bu_init: numpy array (maxUserId) of bu initialization values
    bi_init: numpy array (maxItemId) of bi initialization values
    p_init: numpy array (maxUserId, n_factors) of p initialization values
    t_init: numpy array (maxItemId, n_factors) of t initialization values

  OUTPUT:
    e_arr: numpy array (1D) of MAE for each epoch
    mu: double of mean ratting
    bu_arr: numpy array (maxUserId) of best user baseline
    bi_arr: numpy array (maxItemId) of best item baseline
    p_arr: numpy array (maxUserId, n_factors) of best user preferences p
    t_arr: numpy array (maxItemId, n_factors) of best item topics t

  '''

  # Initializes arrays of users and items as well as numbers for
  # number of ratings, users, and items
  users = np.sort(df_data.userId.unique())
  items = np.sort(df_data.movieId.unique())
  n_ratings = len(df_data)
  n_users = len(users)
  n_items = len(items)

  # Randomly initialize the user baseline, item baseline, user 
  # preferences matrix, and item topic matrix
  mu = np.mean(df_data['rating'])
  if bu_init is None:
    bu = np.random.normal(0, epsilon, n_users)
  else:
    bu = bu_init[users]
  if bi_init is None:
    bi = np.random.normal(0, epsilon, n_items)
  else:
    bi = bi_init[items]
  if p_init is None:
    p = np.random.normal(0, epsilon, (n_users, n_factors))
  else:
    p = p_init[users]
  if t_init is None:
    t = np.random.normal(0, epsilon, (n_items, n_factors))
  else:
    t = t_init[items]

  # Initializes array to convert between userId and uth row in p
  # and itemId and ith row of t. Must be done because non-consecutive
  # userId and itemId's.
  if maxItemId < np.max(items):
    maxItemId = np.max(items)
  if maxUserId < np.max(users):
    maxUserId = np.max(users)
  items_to_i = np.zeros(maxItemId+1,dtype=int)
  items_to_i[items] = np.arange(n_items)
  users_to_u = np.zeros(maxUserId+1,dtype=int)
  users_to_u[users] = np.arange(n_users)

  # Initializes array such that:
  # num_U[i] = number of users rating item i
  # num_I[u] = number of items rated by user u
  num_U = df_data.groupby('movieId').count()['userId'].values
  num_I = df_data.groupby('userId').count()['movieId'].values

  # Intializes c++ arrays for storing dataframe
  userIds = np.zeros(n_ratings, dtype=np.int)
  movieIds = np.zeros(n_ratings, dtype=np.int)
  ratings = np.zeros(n_ratings, dtype=np.double)

  # Initializes update vectors for preferences and topics
  dp = np.zeros(n_factors, dtype=np.double)
  dt = np.zeros(n_factors, dtype=np.double)

  # Initializes MAE array
  e_arr = []

  # Optimization procedure
  epoch = 0
  e_prev = 0               # previous MAE
  de = e_tol + 1           # difference in MAE
  while epoch < n_epochs and de > e_tol:

    # randomize the order of df_data
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    userIds = df_data['userId'].values
    movieIds = df_data['movieId'].values
    ratings = df_data['rating'].values

    e_avg = 0.0
    for idx_rating in range(n_ratings):

      # Finds rows of p and t corresponding to userId and movieId (u and i)
      userId = userIds[idx_rating]
      movieId = movieIds[idx_rating]
      r_ui = ratings[idx_rating]
      u = users_to_u[userId]
      i = items_to_i[movieId]
      n_Iu = num_I[u]
      n_Ui = num_U[i]

      # Calculates error
      p_u_dot_t_i = np.dot(p[u],t[i])
      r_pred_ui = mu+bu[u]+bi[i]+p_u_dot_t_i
      e_ui = r_ui - r_pred_ui

      # Differential latent vectors p_u and t_i and bias terms
      dbu = alpha*(e_ui-lam_u*bu[u]/n_Iu)
      dbi = alpha*(e_ui-lam_i*bi[i]/n_Ui)
      dp = alpha*(e_ui*t[i]-lam_p*p[u]/n_Iu)
      dt = alpha*(e_ui*p[u]-lam_t*t[i]/n_Ui)

      # Updates latent vectors and bias terms
      bu[u] += dbu
      bi[i] += dbi
      p[u] += dp
      t[i] += dt

      # sums absolute errors
      e_avg += abs(e_ui)

    # Finds average error for epoch
    e_avg = e_avg/float(n_ratings)
    e_arr.append(e_avg)

    # Increments de and epoch
    if epoch > 0:
      de = abs(e_avg - e_prev)
    e_prev = e_avg
    epoch += 1

  e_arr = np.array(e_arr)
  bu_arr = np.zeros(maxUserId+1)
  bu_arr[users] = bu
  bi_arr = np.zeros(maxItemId+1)
  bi_arr[items] = bi
  p_arr = np.zeros((maxUserId+1, n_factors))
  p_arr[users] = p
  t_arr = np.zeros((maxItemId+1, n_factors))
  t_arr[items] = t


  return (e_arr, mu, bu_arr, bi_arr, p_arr, t_arr)


@cython.cdivision(True)    # turn off nan check in division
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for function
def SVD_ALS(df_data,
  int n_factors=1, int maxUserId=-1, int maxItemId=-1,
  int n_epochs=40, double e_tol=0.0, double epsilon=0.1,
  double lam_p=0.001, double lam_t=0.001,
  p_init=None, t_init=None):
  '''
  ALS

  Performs Stochastic Gradient Descent to find an approximate solution
  to the SVD problem. Does not find true SVD because regularization used.

  PARAMETER:
    df_data: pandas dataframe with 'userId', 'movieId', and 'rating' 
        columns
    n_factors: positive integer for number of factors to keep
    maxUserId: positive integer for maximum user ID in set
    maxItemId: positive integer for maximum item ID in set
    n_epochs: positive integer for maximum loops through ratings set
    e_tol: positive double for error tolerance (stops early if 
      two consecutive epochs differ by less than e_tol)
    epsilon: small positive double setting the initialized randomness
    lam_p: positive double for p regularization
    lam_t: positive double for t regularization
    p_init: numpy array (maxUserId, n_factors) of p initialization values
    t_init: numpy array (maxItemId, n_factors) of t initialization values

  OUTPUT:
    e_arr: numpy array (1D) of MAE for each epoch
    mu: double mean rating
    p_arr: numpy array (maxUserId, n_factors) of best user preferences p
    t_arr: numpy array (maxItemId, n_factors) of best item topics t

  '''

  # Declarations for ints and doubles
  cdef int epoch, idx_rating, n_ratings             # loop vars
  cdef int userId, movieId, u, i, factor            # id vars
  cdef int n_users, n_items                         # N u's and i's
  cdef double n_Iu, n_Ui                            # N i's rated by u
                                                    # N u's rating i
  cdef double r_ui, r_pred_ui, e_ui, e_avg          # rating, pred, err
  cdef double e_prev, de                            # MAE prev/ d(MAE)

  # Declarations of int and double ndarrays
  cdef np.ndarray[np.int_t, ndim=1] users, items    # user/item arrs
  cdef np.ndarray[np.int_t, ndim=1] items_to_i, users_to_u
                                                    # conversion arrs
  cdef np.ndarray[np.int_t,ndim=1] num_U, num_I     # n Users/Items
  cdef np.ndarray[np.double_t, ndim=2] p, t         # prefs/topics
  cdef np.ndarray[np.int_t, ndim=1] userIds, itemIds# user/item id arr
  cdef np.ndarray[np.double_t, ndim=1] ratings      # rating arr

  # Initializes arrays of users and items as well as numbers for
  # number of ratings, users, and items
  users = np.sort(df_data.userId.unique())
  items = np.sort(df_data.movieId.unique())
  n_ratings = len(df_data)
  n_users = len(users)
  n_items = len(items)

  # Randomly initialize the user 
  # preferences matrix, and item topic matrix
  if p_init is None:
    p = np.random.normal(0, epsilon, (n_users, n_factors))
  else:
    p = p_init[users]
  if t_init is None:
    t = np.random.normal(0, epsilon, (n_items, n_factors))
  else:
    t = t_init[items]
  identity = np.identity(n_factors)
  lam_p_mat = lam_p * identity
  lam_t_mat = lam_t * identity

  # Initializes array to convert between userId and uth row in p
  # and itemId and ith row of t. Must be done because non-consecutive
  # userId and itemId's.
  if maxItemId < np.max(items):
    maxItemId = np.max(items)
  if maxUserId < np.max(users):
    maxUserId = np.max(users)
  items_to_i = np.zeros(maxItemId+1,dtype=int)
  items_to_i[items] = np.arange(n_items)
  users_to_u = np.zeros(maxUserId+1,dtype=int)
  users_to_u[users] = np.arange(n_users)

  # Initializes baseline
  mu = np.mean(df_data['rating'])
  df_baseline = pd.DataFrame(df_data)
  df_baseline.loc[:,'temp'] = df_data.loc[:,'rating']-mu
  df_bu = df_baseline.groupby('userId', as_index=False).temp.mean()
  df_bu.columns = ['userId', 'bu']
  df_baseline = df_baseline.merge(df_bu, on='userId')
  df_baseline.loc[:,'temp'] = df_baseline['rating']-df_baseline['bu']-mu
  df_bi = df_baseline.groupby('movieId', as_index=False).temp.mean()
  df_bi.columns = ['movieId', 'bi']
  df_baseline = df_baseline.merge(df_bi, on='movieId')
  df_baseline.drop(['temp'], axis=1);

  # Intializes c++ arrays for storing dataframe
  userIds = df_baseline['userId'].values
  movieIds = df_baseline['movieId'].values
  ratings = (df_baseline['rating'] - mu - df_baseline['bu'] - df_baseline['bi']).values
  uIds = users_to_u[userIds]
  iIds = items_to_i[movieIds]
  R_csc = csc_matrix((ratings, (uIds, iIds)))
  R_csr = csr_matrix((ratings, (uIds, iIds)))

  # Initializes MAE array
  e_arr = []

  # Optimization procedure
  epoch = 0
  e_prev = 0               # previous MAE
  de = e_tol + 1           # difference in MAE
  while epoch < n_epochs and de > e_tol:

    # p update matrix
    RT = R_csr.dot(t)
    for u in range(n_users):
      Ru = R_csr[u]
      items_rated_by_user = Ru.nonzero()[1]
      Tu = t[items_rated_by_user]
      alpha = np.dot(np.transpose(Tu),Tu) + lam_p_mat
      beta = RT[u]
      p[u] = np.linalg.solve(alpha,beta)

    # t update matrix
    RP = (R_csc.T).dot(p)
    for i in range(n_items):
      Ri = R_csc[:,i]
      users_who_rated_item = Ri.nonzero()[0]
      Pu = p[users_who_rated_item]
      alpha = np.dot(np.transpose(Pu),Pu) + lam_t_mat
      beta = RP[i]
      t[i] = np.linalg.solve(alpha,beta)

    # Finds average error for epoch
    e_avg = 0
    for idx in range(len(ratings)):
      e_avg += abs(
        ratings[idx] - np.dot(
        p[uIds[idx]], t[iIds[idx]]))
    e_avg = e_avg / len(ratings)
    e_arr.append(e_avg)

    # Increments de and epoch
    if epoch > 3:
      de = abs(e_avg - e_prev)
    e_prev = e_avg
    epoch += 1

  e_arr = np.array(e_arr)
  bu_arr = np.zeros(maxUserId+1)
  bu_arr[df_bu['userId']] = df_bu['bu']
  bi_arr = np.zeros(maxItemId+1)
  bi_arr[df_bi['movieId']] = df_bi['bi']
  p_arr = np.zeros((maxUserId+1, n_factors))
  p_arr[users] = p
  t_arr = np.zeros((maxItemId+1, n_factors))
  t_arr[items] = t


  return (e_arr, mu, bu_arr, bi_arr, p_arr, t_arr)
