library(readr)
library(class)
library(gmodels)
library(e1071)
library(C50)

# Wczytanie danych
data_raw <- read_csv("diabetes.csv")
View(data_raw)
str(data_raw)
summary(data_raw)


RNGversion("3.5.2")
set.seed(123)
data_raw <- data_raw[sample(nrow(data_raw)), ]


# Liczba zer w każdej z tych kolumn
sapply(data_raw[-9], function(x) sum(x == 0))

# Lista kolumn do usunięcia zerowych wartości
cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

data <- data_raw[apply(data_raw[cols], 1, function(x) all(x != 0)), ]
sapply(data[-9], function(x) sum(x == 0))



data$Outcome <- factor(data$Outcome, levels = c(0, 1), labels = c("no", "yes"))


round(prop.table(table(data$Outcome)) * 100, digits = 1)
summary(data)

#K-NN (wybieranie najlepszego)


# Normalizacja (min-max)
minmax_normalize <- function(x, min_val, max_val) {
  return((x - min_val) / (max_val - min_val))
}

# Testowane wartości k
# sqrt(313) = 18
k_values <- c(5, 7, 11, 18, 24, 26)

data_train_raw <- data[1:313, 1:8]
data_test_raw  <- data[314:392, 1:8]



####################################### Normalizacja danych (min-max)##################################



# Oblicz min i max ze zbioru treningowego
min_vals <- sapply(data_train_raw, min)
max_vals <- sapply(data_train_raw, max)

# Normalizacja danych treningowych
data_train <- as.data.frame(mapply(minmax_normalize, x=data_train_raw, 
                                   min_val = min_vals, 
                                   max_val = max_vals, SIMPLIFY=FALSE
                                   ))

# Normalizacja danych testowych z użyciem min/max ze zbioru treningowego
data_test <- as.data.frame(mapply(minmax_normalize, data_test_raw, 
                                  min_val = min_vals, 
                                  max_val = max_vals, SIMPLIFY=FALSE
                                  ))

data_train_labels <- data$Outcome[1:313]
data_test_labels  <- data$Outcome[314:392]


results <- data.frame(k = integer(), accuracy = numeric())

for (k in k_values) {
  set.seed(123) 
  
  pred <- knn(train = data_train, test = data_test, cl = data_train_labels, k = k)
  
  acc <- sum(pred == data_test_labels) / length(data_test_labels)
  
  results <- rbind(results, data.frame(k = k, accuracy = acc))
  
  cat("\nk =", k, "| Dokładność:", round(acc * 100, 2), "%\n")
  CrossTable(x = data_test_labels, y = pred, prop.chisq = FALSE, dnn = c("actual", "predicted"))
}

# Najlepszy wynik
results <- results[order(-results$accuracy), ]
best_k <- results$k[1]
best_acc <- results$accuracy[1]

cat("\nNajlepszy k:", best_k, "z dokładnością:", round(best_acc * 100, 2), "%\n")



############################################ STANDARYZACJA ###################################

# Wektor średnich i odchyleń standardowych 
train_means <- colMeans(data_train_raw)
train_sd <- apply(data_train_raw, 2, sd)

#Standaryzacja zbioru treningowego

data_train <- as.data.frame(
  sweep(
    sweep(data_train_raw, 2, train_means, FUN = "-"),   
    2, train_sd, FUN = "/"          
  )
)

# Standaryzacja zbioru testowego 

data_test <- as.data.frame(
  sweep(
    sweep(data_test_raw, 2, train_means, FUN = "-"),    
    2, train_sd,FUN = "/"          
  )
)

results <- data.frame(k = integer(), accuracy = numeric())

for (k in k_values) {
  set.seed(123) 
  
  pred <- knn(train = data_train, test = data_test, cl = data_train_labels, k = k)
  
  acc <- sum(pred == data_test_labels) / length(data_test_labels)
  
  results <- rbind(results, data.frame(k = k, accuracy = acc))
  
  cat("\nk =", k, "| Dokładność:", round(acc * 100, 2), "%\n")
  CrossTable(x = data_test_labels, y = pred, prop.chisq = FALSE, dnn = c("actual", "predicted"))
}

# Najlepszy wynik
results <- results[order(-results$accuracy), ]
best_k <- results$k[1]
best_acc <- results$accuracy[1]

cat("\nNajlepszy k:", best_k, "z dokładnością:", round(best_acc * 100, 2), "%\n")



# NAIVE BAYES

#min/max
data_train <- as.data.frame(mapply(minmax_normalize, x=data_train_raw, 
                                   min_val = min_vals, 
                                   max_val = max_vals, SIMPLIFY=FALSE
))

# Normalizacja danych testowych z użyciem min/max ze zbioru treningowego
data_test <- as.data.frame(mapply(minmax_normalize, data_test_raw, 
                                  min_val = min_vals, 
                                  max_val = max_vals, SIMPLIFY=FALSE
))


data_classifier <- naiveBayes(data_train, data_train_labels)

data_test_pred_B <- predict(data_classifier,data_test)

CrossTable(data_test_pred_B, data_test_labels, prop.chisq = FALSE, 
           prop.c = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))


# STANDARYZACJA

#Standaryzacja zbioru treningowego

data_train <- as.data.frame(
  sweep(
    sweep(data_train_raw, 2, train_means, FUN = "-"),   
    2, train_sd, FUN = "/"          
  )
)

# Standaryzacja zbioru testowego 

data_test <- as.data.frame(
  sweep(
    sweep(data_test_raw, 2, train_means, FUN = "-"),    
    2, train_sd,FUN = "/"          
  )
)

data_classifier <- naiveBayes(data_train, data_train_labels)

data_test_pred_B <- predict(data_classifier,data_test)

CrossTable(data_test_pred_B, data_test_labels, prop.chisq = FALSE, 
           prop.c = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

#Bez normalizacji 

# Podział danych
data_train <- data[1:313, 1:8]
data_test  <- data[314:392, 1:8]

data_classifier <- naiveBayes(data_train, data_train_labels)

data_test_pred_B <- predict(data_classifier,data_test)

CrossTable(data_test_pred_B, data_test_labels, prop.chisq = FALSE, 
           prop.c = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))


# Obliczenie dokładności
sum(data_test_pred_B == data_test_labels) / length(data_test_labels)

# DRZEWA DECYZYJNE 
set.seed(123)
# Podział danych

data_train_d <- data[1:313, ]
data_test_d  <- data[314:392,]

d_model <- C5.0(data_train_d[-9], data_train_d$Outcome)

d_model

summary(d_model)

# na zbiorze testowym 

d_model_pred <- predict(d_model, data_test_d)


CrossTable(data_test_d$Outcome, d_model_pred, prop.chisq = FALSE,
           prop.c = FALSE, prop.r = FALSE, dnn = c('actual', 'predicted'))

# Poprawa modelu 
set.seed(123)
d_model_boost10 <- C5.0(data_train_d[-9], data_train_d$Outcome, trails = 10)

d_model_boost10
summary(d_model_boost10)


d_model_pred10 <- predict(d_model_boost10, data_test_d)


CrossTable(data_test_d$Outcome, d_model_pred10, prop.chisq = FALSE,
           prop.c = FALSE, prop.r = FALSE, dnn = c('actual', 'predicted'))

matrix_dim <- list(c('no', 'yes'), c('no', 'yes'))
names(matrix_dim) <- c('predicted', 'actual')

error_cost <- matrix(c(0,1,5,0), nrow = 2, dimnames = matrix_dim)
error_cost

d_model_cost <- d_model_boost10 <- C5.0(data_train_d[-9], data_train_d$Outcome, costs=error_cost)
d_model_cost_pred <- predict(d_model_cost, data_test_d)

CrossTable(data_test_d$Outcome, d_model_cost_pred, prop.chisq = FALSE,
           prop.c = FALSE, prop.r = FALSE, dnn = c('actual', 'predicted'))
