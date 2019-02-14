
library("dplyr")
library("mice")
library("mi")
library("missForest")
library("betareg")
library("ggplot2")
library("caret")
library("keras")


###### GETTING THE DATASET
df <- read.csv("data//ann-train.data", sep = " ", header = F)


# ----------------------------------------------------------------
# -------------------- F U N C T I O N S -------------------------
# ----------------------------------------------------------------

generate_data <- function(model_name, df, perc, rr, target_var, positive){
  cat("generating data", fill = T)
  index <- which(names(df) == eval(target_var))
  pattern <- rep(1, length(names(df)))
  pattern[index] <- 0
  amp <- ampute(df, prop = perc, mech = "MAR", patterns = pattern)
  miss <- ifelse(is.na(amp$amp[, eval(target_var)]), 1, 0)

  # MULTIPLE IMPUTATIONS WITH MI PACKAGE
  if(model_name == "mi"){
    df_covariates_mi <- df
    df_covariates_mi[, eval(target_var)][miss == 1] <- NA

    mdf <- mi::missing_data.frame(df_covariates_mi)
    options(mc.cores = 2)
    cat(model_name, " imputations", fill = T)
    imputations <- mi::mi(mdf, n.iter = 5, n.chains = 4, max.minutes = 20)

    dfs <- mi::complete(imputations)
    cat(model_name, " imputations completed", fill = T)

    mi <- dfs$`chain:1`
    mi <- mi[mi[, ncol(mi)] == T, index]
  }
   
  # MULTIPLE IMPUTATIONS WITH MICE PACKAGE
  if(model_name == "mice"){
    df_covariates_mi <- df
    df_covariates_mi[, eval(target_var)][miss == 1] <- NA

    cat(model_name, " imputations", fill = T)
    imp <- mice(df_covariates_mi, method = "logreg", m = 1, maxit = 1)
    imputations <- mice::complete(imp)
    cat(model_name, " imputations completed", fill = T)
    
    mi <- imputations[miss == 1, index]
  }

  # MULTIPLE IMPUTATIONS WITH MICE PACKAGE USING RANDOMFOREST
  if(model_name == "missForest"){
    df_covariates_mi <- df
    df_covariates_mi[, eval(target_var)][miss == 1] <- NA
    
    cat(model_name, " imputations", fill = T)
    
    imp <- missForest::missForest(df_covariates_mi, verbose = FALSE)
    imputations <- imp$ximp
    cat(model_name, " imputations completed", fill = T)
    
    mi <- imputations[miss == 1, index]
  }

  # MULTIPLE IMPUTATIONS WITH KERAS PACKAGE USING DNN
  if(model_name == "DNN"){
    df_covariates_scaled <- df %>% select(-c(eval(target_var)))
    for(uu in 1:ncol(df_covariates_scaled)){
      df_covariates_scaled[, uu] <- as.numeric(df_covariates_scaled[, uu])
      df_covariates_scaled[, uu] <- scale(df_covariates_scaled[, uu], center = T, scale = sd(df_covariates_scaled[, uu]))
    }
    df_covariates_scaled <- as.matrix(data.matrix(df_covariates_scaled))
    
    train_X <- df_covariates_scaled[miss == 0, 1:dim(df_covariates_scaled)[2]]
    test_X <- df_covariates_scaled[miss == 1, 1:dim(df_covariates_scaled)[2]]
    
    train_Y <- as.numeric(df[miss == 0, index]) - 1
    test_Y <- as.numeric(df[miss == 1, index]) - 1
    
    cat("data generated", fill = T)
    cat(model_name, "inizialization", fill = T)

    input <- layer_input(shape = c(ncol(df) - 1), name = 'covariates_input')
    encoded_input <- input %>%
      layer_dense(units = n_units, activation = 'relu') %>%
      layer_batch_normalization()
    output_layer <- encoded_input %>% 
      layer_dense(units = 1, activation = "sigmoid")
    model <- keras_model(input, output_layer)
    model %>% compile(
      optimizer = "rmsprop",
      loss = "binary_crossentropy",
      metrics = c("acc")
    )
    
    cat(model_name, "training", fill = T)
    history <- model %>% fit(
      train_X, train_Y,
      epochs = n_epochs,
      batch_size = n_batch,
      validation_split = n_split,
      callbacks = callbacks
    )
    
    cat(model_name, "prediction", fill = T)
    preds <- model %>% predict(test_X)
    result <- ifelse(preds > 0.5, 1, 0)
    cat(model_name, " imputations completed", fill = T)

    mi <- factor(result)
    ll <- levels(df[, eval(target_var)])
    levels(mi) <- ll
  }
  
  label <- df[miss == 1, index]
  try(cm_mi <- confusionMatrix(as.factor(mi), as.factor(label), positive = positive))
  
  cat(model_name, "saving data", fill = T)
  name <- paste0("CM_r_", rr, "_", model_name, "_", perc)
  assign(name, cm_mi, envir = .GlobalEnv)
  cat(model_name, "ALL DONE", fill = T)
}

test_increasing <- function(perc, rr, models, df, target_var, positive){
  cat("============================================================", fill = T)
  cat("Round N", rr, " \t Missing  = " , perc * 100, "%", fill = T)
  df <- df
  for(model_name in models){
    cat(model_name, fill = T)
    generate_data(model_name, df = df, perc = perc, rr = rr, target_var = target_var, positive = positive)
  }
}

summarise_SE <- function(df, conf_interval, groupings, statistic){
  qstatistic <- enquo(statistic)
  tmp <- df %>% group_by(!!!groupings) %>% 
    summarise(N = n(), mean = mean(!!qstatistic, na.rm = T), sd = sd(!!qstatistic, na.rm = T))
  tmp$se <- tmp$sd / sqrt(tmp$N)
  tmp$ci <- tmp$se * qt(conf_interval / 2 + .5, tmp$N - 1)
  names(tmp)
  tmp
}

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------


###### DATA PREPROCESSING
df <- data.frame(df)
df <- df[, 1:22]

# DNN VARS
n_units <- round((ncol(df) - 1 ) / 2 ) # n units dense layer
n_epochs <- 20
n_batch <- 64
n_split <- 0.3
patience_lr <- 2
patience_stop <- 6
dr <- 0.1
l2 <- 0.0001

callbacks = list(
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.2,
    patience = patience_lr,
    verbose = 1
  ),
  callback_early_stopping(
    monitor = "val_loss",
    patience = patience_stop)
)

###### IMPUTATING MISSING DATA
models <- c("mi", "mice", "missForest", "DNN")
repeats <- 10
probs <- c(1:5)/10
target <- "V3"

df[, eval(target)] <- as.factor(df[, eval(target)])
positive <- levels(df[, eval(target)])[2]

for(i in 1:repeats){
  for(p in probs){
    test_increasing(p, rr = i, models, df, target_var = target, positive = positive)
  }
}

###### OOB MISCLASSIFICATION ERROR %
res <- data.frame()
for(i in 1:repeats){
  for(p in probs){
    for(model_name in models){
      name <- paste0("CM_r_", i, "_", model_name, "_", p)
      cm <- get(name)$table
      oob_err <- (cm[1,2] + cm[2,1]) / sum(cm)
      tmp <- data.frame(package = model_name, missing = p, rep = i, oob_err = oob_err)
      res <- rbind(res, tmp)
    }
  }
}

write.csv(res, "results.csv")

grouping <- quos(package, missing)
oob <- summarise_SE(df = res, .95, statistic = oob_err, grouping = grouping)

######  GENERATING PLOT
pd <- 0.09
p_oob <- ggplot(oob, aes(x=missing, y=mean, fill=package)) + 
  geom_bar(position=position_dodge(pd), stat="identity", size=.1) + 
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se),
                size=.75,
                width=.02, position = position_dodge(pd)) +
  guides(fill = guide_legend(title =  element_blank(), ncol = 8)) +
  theme(legend.text = element_text(size = 10)) +
  xlab("Missing data (%)") +
  ylab("OOB error (avg + se)") +
  labs(title="OOB misclassification error (%)") +
  expand_limits(y=c(0.0, 0.005)) +
  scale_y_continuous(breaks=0:10*0.1) +
  scale_x_continuous(breaks=0:10*0.1) +
  theme(legend.position="bottom")

p_oob
