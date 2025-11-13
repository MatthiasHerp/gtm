##### Generate Train and Test Data as well as Likelihood for the Simulation Study #####

library(VineCopula)

get_cdf_values_n <- function(data, means, vars) { 
  pnorm(data, mean = 0, sd = 1)
  #apply(data, 2, function(x) pnorm(x, mean = means, sd = sqrt(vars))) 
}

compute_nD_log_likelihood <- function(data, RVM, means = c(rep(0, ncol(data))), vars = rep(1, ncol(data))) { 
    ll <- RVineLogLik(pnorm(data, mean = 0, sd = 1), RVM, separate=TRUE)$loglik
     
    for (col_num in 1:dim(data)[2] ){
      ll <- ll + dnorm(data[,col_num], mean = 0, sd = 1, log = TRUE)
    }
    
    ll
  
}

create_cvine_structure_matrix <- function(D) {
  # Initialize a D x D matrix filled with zeros
  M <- matrix(0, nrow = D, ncol = D)
  
  # Fill the main diagonal
  #diag(A) <- D:1
  
  # Fill the subdiagonals
  for (row in 1:D) {
    for (col in 1:row) {
      if(col==row){
        M[row,col] <- D-row+1
      #} else if(col==(row-1)){
      #  M[row,col] <- 1
      } else {
        M[row,col] <- D-row+1 #D-row+2
      }
    }
  }
  return(M)
}


create_copula_families_matrix <- function(D, I_percent=0.2, Independence_Tree=2, gauss_only=FALSE) {
  # If Tree Number "Independence_Tree" is Independent then 
  # all Rows smaller than "Independence_Row" have independence Copulas
  Independence_Row <- D - Independence_Tree + 2
  
  # Initialize a DxD matrix with zeros
  mat <- matrix(0, nrow = D, ncol = D)
  
  # Note: 
  # - We have only copulas wher we can set the parameters with kendas tau 
  #   to ensure we dont have to extreme dependencies dominating data,
  #   this means no BB copulas
  # - We have included the Gauss (1) and the T-Copula (2) 4 times to 
  #    compensate for the four rotations of the other copulas
  # - Same for the Frank Copula which we also have in 4 times
  all_dependent_copula_families <- c(1, 2, 
                                     1, 2,
                                     1, 2,
                                     1, 2,
                                     5, 5,
                                     5, 5,
                                     3, 4, 6, #7, 8, 9, 10, 
                                     13, 14, 16, #18, 
                                     23, 24, 26, #27, 28, 29, 30, 
                                     33, 34, 36)#, 37, 38, 39, 40) 
                                     #, 104, 114, 124, 134, 204, 214, 224, 234)
  
  if (gauss_only == TRUE){
    all_dependent_copula_families <- c(1,1,1,1)
  }
  
  num_copulas <- D*(D-1)/2
  copula_families_lower_rows <- sample(all_dependent_copula_families, num_copulas, replace = TRUE)
  
  num_independence_copulas <- round(num_copulas * I_percent)
  num_dependence_copulas <- num_copulas - num_independence_copulas
  
  # add proportion of independencies
  copula_families_lower_rows[(num_dependence_copulas+1):length(copula_families_lower_rows)] <- 0
  
  # Shuffle the vector
  copula_families_lower_rows <- sample(copula_families_lower_rows)
  
  # Fill in the matrix according to the specified pattern
  copula_number <- 1
  for (row in 1:D) {
    for (col in 1:D) {
      if (col < row) { # no diagonal elements
        if (row < Independence_Row ){ 
          mat[row, col] <- 0 
          copula_number <- copula_number + 1 
        } else {
          mat[row, col] <- copula_families_lower_rows[copula_number]
          copula_number <- copula_number + 1 
        }
      }
    }
  }
  
  return(mat)
}




create_copula_par_matrix <- function(D, family, par2=FALSE, mintau=0.3, maxtau=0.7) {
  mat <- family
  
   
  # Fill in the matrix according to the specified pattern
  for (row in 1:D) {
    for (col in 1:D) {
      if (col < row) { # no diagonal elements
        if (family[row, col] %in% c(1) ){ #Gauss which can be positive or negative
          mat[row, col] <- runif(1, min = BiCopTau2Par(1,mintau), max = BiCopTau2Par(1,maxtau)) * sample(c(1,-1),1)
          
        } else if (family[row, col] %in% c(2)){ #T which can be positive or negative
          mat[row, col] <- runif(1, min = BiCopTau2Par(2,mintau), max = BiCopTau2Par(2,maxtau)) * sample(c(1,-1),1)
        
        } else if (family[row, col] %in% c(3,13,23,33)){ #Clayton (unequal rotated Clayton)
          mat[row, col] <- runif(1, min = BiCopTau2Par(3,mintau), max = BiCopTau2Par(3,maxtau))
          
        } else if (family[row, col] %in% c(4,14,24,34)){ #Gumbel
          mat[row, col] <- runif(1, min = BiCopTau2Par(4,mintau), max = BiCopTau2Par(4,maxtau))
          
        } else if (family[row, col] %in% c(5) ){ #Frank which can be positive or negative
          mat[row, col] <- runif(1, min = BiCopTau2Par(5,mintau), max = BiCopTau2Par(5,maxtau)) * sample(c(1,-1),1)
          
        } else if (family[row, col] %in% c(6,16,26,36) ){ #joe
          mat[row, col] <- runif(1, min = BiCopTau2Par(6,mintau), max = BiCopTau2Par(6,maxtau))
        }
        
        if (family[row, col] %in% c(23,24,26,
                                 33,34,36) ){ #for negative correlation
          mat[row, col] <- -1 * mat[row, col]
          
        }
        
        if (par2==TRUE && family[row, col]==2){
          mat[row, col] <- sample(3:10,1)
        }
      }
    }
  }
  
  return(mat)
}

create_copula_kendall_tau_matrix <- function(par, family, par2=FALSE) {
  mat <- family
  D <- nrow(family)
  
  
  # Fill in the matrix according to the specified pattern
  for (row in 1:D) {
    for (col in 1:D) {
      if (col < row) { # no diagonal elements
        mat[row, col] <- BiCopPar2Tau(family[row, col],par[row, col])
      }
    }
  }
  
  return(mat)
}

create_sim_study_data <- function(path_string,
                                  N_train = 4000,
                                  D_dimensions = 10,
                                  I_percent = 0.1,
                                  Independence_Tree = 2,
                                  num_seeds = 1,
                                  grid_obs_num = 40000,
                                  folder="cvinet1_10_4000",
                                  gauss_only=FALSE){
  
  directory <- paste0(path_string,folder)
  if (!dir.exists(directory)) {
    dir.create(directory, recursive = TRUE)
  }
  setwd(directory)
  
  # Seeds
  seeds <- 1:num_seeds

  # For each seed we sample, compute sample, log likelihoods, estimate the model on the sample and comptue est log likelihoods
  for (seed_value in seeds){
    set.seed(seed_value)
    
    # define 3-dimensional R-vine tree structure matrix
    #Matrix <- c(1, 3, 2,
    #            0, 3, 2,
    #            0, 0, 2)
    #Matrix <- matrix(Matrix, 3, 3)
    rvinematrix <- create_cvine_structure_matrix(D_dimensions)
    
    #print(rvinematrix)
    
    RVineMatrixCheck(rvinematrix)
    
    # define R-vine pair-copula family matrix
    #family <- c(0, 0, 6,
    #            0, 0, 6,
    #            0, 0, 0)
    #family <- matrix(family, 3, 3)
    family <- create_copula_families_matrix(D = D_dimensions,
                                            I_percent = I_percent,
                                            Independence_Tree=Independence_Tree,
                                            gauss_only=gauss_only)
    
    # define R-vine pair-copula parameter matrix
    #par <- c(0, 0, 4,
    #         0, 0, 3,
    #         0, 0, 0)
    #par <- matrix(par, 3, 3)
    par <- create_copula_par_matrix(D_dimensions, family)
    # define second R-vine pair-copula parameter matrix
    #par2 <- matrix(0, 3, 3)
    par2 <- create_copula_par_matrix(D_dimensions, family, par2=TRUE)
    
    ## define RVineMatrix object
    RVM <- RVineMatrix(Matrix = rvinematrix, family = family,
                       par = par, par2 = par2,
                       #names = c("y_1","y_2","y_3")
    )
    
    write.table(rvinematrix,paste0(seed_value,"_rvinematrix.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(family,paste0(seed_value,"_family.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(par,paste0(seed_value,"_par.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(par2,paste0(seed_value,"_par2.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    
    kendall_tau_matrix <- create_copula_kendall_tau_matrix(par, family, par2=FALSE)
    write.table(kendall_tau_matrix,paste0(seed_value,"_kendalltaumatrix.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    
    test_data_copula <- RVineSim(grid_obs_num, RVM)
    grid_test <- qnorm(test_data_copula, mean = 0, sd = 1)
    
    test_log_likelihoods <- compute_nD_log_likelihood(grid_test, RVM)
    
    print("defined the copula")
    
    write.table(grid_test,paste0(seed_value,"_grid_test.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(test_log_likelihoods,paste0(seed_value,"_test_log_likelihoods.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    
    print("generated test data")
    

    sample_cdf_train <- RVineSim(N_train, RVM)
    sample_train <- qnorm(sample_cdf_train,mean=0, sd = 1, lower.tail = TRUE, log.p = FALSE)
    train_log_likelihood <- compute_nD_log_likelihood(sample_train, RVM)

    sample_cdf_validate <- RVineSim(N_train, RVM)
    sample_validate <- qnorm(sample_cdf_validate,mean=0, sd = 1, lower.tail = TRUE, log.p = FALSE)
    validate_log_likelihood <- compute_nD_log_likelihood(sample_validate, RVM)

    write.table(sample_train,paste0(seed_value,"_sample_train.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(sample_validate,paste0(seed_value,"_sample_validate.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(train_log_likelihood,paste0(seed_value,"_train_log_likelihoods.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(validate_log_likelihood,paste0(seed_value,"_validate_likelihoods.csv"), row.names = FALSE, col.names = FALSE, sep=",")


    # Compute Moments
    means <- colMeans(sample_train)
    vars <- diag(cov(sample_train))

    # Estimate cdf values
    sample_est_cdf_train <- pnorm(sample_train, mean = 0, sd = 1)
    
    # Fit Copula Model
    RVM_est <- RVineSeqEst(
      sample_est_cdf_train,
      RVM,
      method = "mle")

    # Compute estimated Log Likelihood
    est_train_log_likelihoods <- compute_nD_log_likelihood(sample_train,RVM_est)
    est_test_log_likelihoods <- compute_nD_log_likelihood(grid_test,RVM_est)

    write.table(est_train_log_likelihoods,paste0(seed_value,"_est_train_log_likelihoods.csv"), row.names = FALSE, col.names = FALSE, sep=",")
    write.table(est_test_log_likelihoods,paste0(seed_value,"_est_test_log_likelihoods.csv"), row.names = FALSE, col.names = FALSE, sep=",")
  }

}

# example path to where to store the folders with the experiment data
path_string <- "/Users/matthiasherp/Desktop/"

create_sim_study_data(path_string = path_string,
                      N_train = 125,
                      D_dimensions = 10,
                      I_percent = 0.1,
                      Independence_Tree = 4,
                      num_seeds = 30,
                      grid_obs_num = 40000,
                      folder="cvine_10_125")

create_sim_study_data(path_string = path_string,
                      N_train = 250,
                      D_dimensions = 10,
                      I_percent = 0.1,
                      Independence_Tree = 4,
                      num_seeds = 30,
                      grid_obs_num = 40000,
                      folder="cvine_10_250")

create_sim_study_data(path_string = path_string,
                      N_train = 500,
                      D_dimensions = 10,
                      I_percent = 0.1,
                      Independence_Tree = 4,
                      num_seeds = 30,
                      grid_obs_num = 40000,
                      folder="cvine_10_500")

create_sim_study_data(path_string = path_string,
                      N_train = 1000,
                      D_dimensions = 10,
                      I_percent = 0.1,
                      Independence_Tree = 4,
                      num_seeds = 30,
                      grid_obs_num = 40000,
                      folder="cvine_10_1000")

create_sim_study_data(path_string = path_string,
                      N_train = 2000,
                      D_dimensions = 10,
                      I_percent = 0.1,
                      Independence_Tree = 4,
                      num_seeds = 30,
                      grid_obs_num = 40000,
                      folder="cvine_10_2000")

create_sim_study_data(path_string = path_string,
                      N_train = 4000,
                      D_dimensions = 10,
                      I_percent = 0.1,
                      Independence_Tree = 4,
                      num_seeds = 30,
                      grid_obs_num = 40000,
                      folder="cvine_10_4000")

