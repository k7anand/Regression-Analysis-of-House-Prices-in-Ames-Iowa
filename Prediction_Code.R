

library(dplyr)
library(Matrix)
library(glmnet)


original_data <- read.csv("C:/Users/asus/Desktop/AmesHousing.csv")


data = original_data

drop_cols <- c("PID", "Utilities", "Street", "Pool.QC", "Misc.Feature", 
               "Condition.2", "Land.Slope", "Roof.Matl")

data <- data[, setdiff(names(data), drop_cols)]
colnames(data)


missing_percent <- sapply(data, function(x) {
  sum(is.na(x)) / nrow(data)
})

missing_percent

# drop columns with more than 20 percent missing values
drop_cols <- names(missing_percent[missing_percent > 0.2])
drop_cols

data <- data[, !(names(data) %in% drop_cols)]


breaks <- seq(1900, 2025, by = 10)

# additional break for values before 1900
breaks <- c(-Inf, breaks)


# create interval labels
labels <- sprintf("%s-%s", head(breaks, -1), tail(breaks - 1, -1))
labels[1] <- "Before 1900"
labels
data$Year.Built <- cut(data$Year.Built, breaks = breaks, labels = labels, right = TRUE)

data$Bedroom.AbvGr <- as.factor(data$Bedroom.AbvGr)
data$TotRms.AbvGrd <- as.factor(data$TotRms.AbvGrd)
data$Fireplaces <- as.factor(data$Fireplaces)
data$Yr.Sold <- as.factor(data$Yr.Sold)
data$MS.SubClass <- as.factor(data$MS.SubClass)
# convert to binary categories
data$Misc.Val <- as.factor(ifelse(data$Misc.Val == 0, 0, 1))
data$Pool.Area <- as.factor(ifelse(data$Pool.Area == 0, 0, 1))

numeric_cols <- sapply(data, is.numeric)
print(numeric_cols)

# handle missing values in numeric columns using interpolation
data[numeric_cols] <- lapply(data[numeric_cols], function(column) {
  if (any(is.na(column))) {
    filled_column <- approx(
      seq_along(column), 
      column, 
      xout = seq_along(column), 
      rule = 2 # Extrapolates values outside the range if there are missing values at the beginning or end of the column
    )$y
    return(filled_column)
  } else {
    return(column)
  }
})

categorical_cols <- sapply(data, is.factor) | sapply(data, is.character)

data[categorical_cols] <- lapply(data[categorical_cols], function(column) {
 
if (any(is.na(column))) {
    mode_value <- names(which.max(table(column)))
    column[is.na(column)] <- mode_value
  }
  return(column)
})


# apply transformations
data$SalePrice <- log(data$SalePrice)
data$Lot.Area <- log(data$Lot.Area)
data$Lot.Frontage <- 1 / data$Lot.Frontage
data$BsmtFin.SF.1 <- log(data$BsmtFin.SF.1)
data$BsmtFin.SF.2 <- log(data$BsmtFin.SF.2)
data$Bsmt.Unf.SF <- sqrt(data$Bsmt.Unf.SF)
data$Total.Bsmt.SF <- log(data$Total.Bsmt.SF)
data$X1st.Flr.SF <- log(data$X1st.Flr.SF)
data$X2nd.Flr.SF <- log(data$X2nd.Flr.SF)
data$Gr.Liv.Area <- sqrt(data$Gr.Liv.Area)
data$Garage.Area <- log(data$Garage.Area)
data$Wood.Deck.SF <- log(data$Wood.Deck.SF)
data$Enclosed.Porch <- log(data$Enclosed.Porch)
data$X3Ssn.Porch <- 1 / data$X3Ssn.Porch
data$Screen.Porch <- log(data$Screen.Porch)


cols_to_check <- c("SalePrice", "Lot.Area", "BsmtFin.SF.1", "BsmtFin.SF.2", 
                   "Total.Bsmt.SF", "X1st.Flr.SF", "X2nd.Flr.SF", "Garage.Area", 
                   "Wood.Deck.SF", "Enclosed.Porch", "Screen.Porch")

# replace negative results with 0
for (col in cols_to_check) {
  data[[col]][data[[col]] < 0] <- 0
  data[[col]][!is.finite(data[[col]])]
}
inverse_cols <- c("Lot.Frontage", "X3Ssn.Porch")
for (col in inverse_cols) {
  data[[col]][!is.finite(data[[col]])] <- 0
}

data$Exter.Qual <- factor(data$Exter.Qual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
data$Kitchen.Qual <- factor(data$Kitchen.Qual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)


# intraction terms
names(data)
data$OverallQual_GrLivArea <- data$Overall.Qual * data$Gr.Liv.Area
data$ExterQual_OverallQual <- as.numeric(data$Exter.Qual) * data$Overall.Qual
data$GarageCars_GarageArea <- data$Garage.Cars * data$Garage.Area
data$TotalBsmtSF_GrLivArea <- data$Total.Bsmt.SF * data$Gr.Liv.Area
data$KitchenQual_OverallQual <- as.numeric(data$Kitchen.Qual) * data$Overall.Qual


numeric_cols <- setdiff(names(data)[sapply(data, is.numeric)], "SalePrice")

# Standardize only numerical predictor columns (exclude "SalePrice")
data[numeric_cols] <- lapply(data[numeric_cols], scale)


response_var <- "SalePrice"

# Retry creating the sparse model matrix
data_features <- data[, !names(data) %in% response_var]
# Create the sparse model matrix without including SalePrice
x <- sparse.model.matrix(~ ., data = data_features)[, -1]
y <- data[[response_var]]
# fit LASSO
set.seed(60) # to ensure reproducibility of random number generation
lasso_model <- cv.glmnet(x, y, alpha = 1, family = "gaussian")
# "gaussian" for continuous response variables (linear regression).

best_lambda <- lasso_model$lambda.min
print(best_lambda)
final_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)





lasso_coefficients <- coef(final_model)
coef_matrix <- as.matrix(lasso_coefficients)

non_zero_coefficients <- coef_matrix[coef_matrix != 0, , drop = FALSE]

print(non_zero_coefficients)

predicted_log_prices <- predict(final_model, x )
predicted_log_prices
# Back-transform predictions and actual values if log transformation was applied
predicted_prices <- exp(predicted_log_prices)
predicted_prices
actual_prices <- exp(y)  # Back-transform the actual values
actual_prices
# Calculate Mean Squared Error (MSE)
mse <- mean((predicted_prices - actual_prices)^2)

# Print the MSE
print(paste("Mean Squared Error:", mse))

par(mfrow = c(2, 2))
plot(final_model)
first_row <- x[1, , drop = FALSE]
# Predict for the first row
predicted_log_price_first_row <- predict(final_model, first_row)
# Convert to original scale
predicted_price_first_row <- exp(predicted_log_price_first_row)
predicted_price_first_row

# Load the necessary library for diagnostic plots
library(ggplot2)

# Generate predictions for the model
y_hat <- predict(final_model, newx = x)

# Calculate residuals
residuals <- y - y_hat

# Combine into a data frame
diagnostics_df <- data.frame(
  Fitted = as.vector(y_hat),
  Residuals = as.vector(residuals)
)

# Set up a 2x2 grid for plots
par(mfrow = c(2, 2))

# 1. Residuals vs Fitted
plot(diagnostics_df$Fitted, diagnostics_df$Residuals,
     main = "Residuals vs Fitted",
     xlab = "Fitted values", ylab = "Residuals",
     pch = 20, col = "black")
abline(h = 0, col = "red", lty = 2)

# 2. Normal Q-Q Plot of Residuals
qqnorm(diagnostics_df$Residuals, main = "Q-Q Residuals", pch = 20, col = "black")
qqline(diagnostics_df$Residuals, col = "red")

# 3. Scale-Location Plot
standardized_residuals <- residuals / sd(residuals)
plot(diagnostics_df$Fitted, sqrt(abs(standardized_residuals)),
     main = "Scale-Location",
     xlab = "Fitted values",
     ylab = expression(sqrt("|Standardized residuals|")),
     pch = 20, col = "black")
abline(h = 0, col = "red", lty = 2)

# 4. Residuals vs Observations Index Plot (as a substitute for Leverage plot)
plot(1:length(standardized_residuals), standardized_residuals,
     main = "Residuals vs Observation Index",
     xlab = "Observation Index",
     ylab = "Standardized Residuals",
     pch = 20, col = "black")
abline(h = 0, col = "red", lty = 2)


# TEST THE MODEL
new_house <- data.frame(
  `PID` = 526301100,
  `MS SubClass` = 20,
  `MS Zoning` = "RL",
  `Lot Frontage` = 500,
  `Lot Area` = 31770,
  `Street` = "Pave",
  `Alley` = NA,
  `Lot Shape` = "IR1",
  `Land Contour` = "Lvl",
  `Utilities` = "AllPub",
  `Lot Config` = "Corner",
  `Land Slope` = "Gtl",
  `Neighborhood` = "NAmes",
  `Condition 1` = "Norm",
  `Condition 2` = "Norm",
  `Bldg Type` = "1Fam",
  `House Style` = "1Story",
  `Overall Qual` = 6,
  `Overall Cond` = 5,
  `Year Built` = 2020,
  `Year Remod/Add` = 2020,
  `Roof Style` = "Hip",
  `Roof Matl` = "CompShg",
  `Exterior 1st` = "BrkFace",
  `Exterior 2nd` = "Plywood",
  `Mas Vnr Type` = "Stone",
  `Mas Vnr Area` = 112.0,
  `Exter Qual` = "TA",
  `Exter Cond` = "TA",
  `Foundation` = "CBlock",
  `Bsmt Qual` = "TA",
  `Bsmt Cond` = "Gd",
  `Bsmt Exposure` = "Gd",
  `BsmtFin Type 1` = "BLQ",
  `BsmtFin SF 1` = 639.0,
  `BsmtFin Type 2` = "Unf",
  `BsmtFin SF 2` = 0.0,
  `Bsmt Unf SF` = 441.0,
  `Total Bsmt SF` = 1080.0,
  `Heating` = "GasA",
  `Heating QC` = "Fa",
  `Central Air` = "Y",
  `Electrical` = "SBrkr",
  `1st Flr SF` = 1656,
  `2nd Flr SF` = 0,
  `Low Qual Fin SF` = 0,
  `Gr Liv Area` = 1656,
  `Bsmt Full Bath` = 1.0,
  `Bsmt Half Bath` = 0.0,
  `Full Bath` = 1,
  `Half Bath` = 0,
  `Bedroom AbvGr` = 3,
  `Kitchen AbvGr` = 1,
  `Kitchen Qual` = "TA",
  `TotRms AbvGrd` = 7,
  `Functional` = "Typ",
  `Fireplaces` = 2,
  `Fireplace Qu` = "Gd",
  `Garage Type` = "Attchd",
  `Garage Yr Blt` = 1960.0,
  `Garage Finish` = "Fin",
  `Garage Cars` = 2.0,
  `Garage Area` = 528.0,
  `Garage Qual` = "TA",
  `Garage Cond` = "TA",
  `Paved Drive` = "P",
  `Wood Deck SF` = 210,
  `Open Porch SF` = 62,
  `Enclosed Porch` = 0,
  `3Ssn Porch` = 0,
  `Screen Porch` = 0,
  `Pool Area` = 0,
  `Pool QC` = NA,
  `Fence` = NA,
  `Misc Feature` = NA,
  `Misc Val` = 0,
  `Mo Sold` = 5,
  `Yr Sold` = 2010,
  `Sale Type` = "WD ",
  `Sale Condition` = "Normal",
  `SalePrice` = 0
)

test <- original_data
test <- rbind(new_house, test)

drop_cols <- c("PID", "Utilities", "Street", "Pool.QC", "Misc.Feature", 
               "Condition.2", "Land.Slope", "Roof.Matl", "Alley", "Fireplace.Qu", "Fence")

test <- test[, !(names(test) %in% drop_cols)]
test
breaks <- seq(1900, 2025, by = 10)
# additional break for values before 1900
breaks <- c(-Inf, breaks)
# create interval labels
labels <- sprintf("%s-%s", head(breaks, -1), tail(breaks - 1, -1))
labels[1] <- "Before 1900"
labels
test$Year.Built <- cut(test$Year.Built, breaks = breaks, labels = labels, right = TRUE)


test[is.na(test)] <- 0

test$Bedroom.AbvGr <- as.factor(test$Bedroom.AbvGr)
test$TotRms.AbvGrd <- as.factor(test$TotRms.AbvGrd)
test$Fireplaces <- as.factor(test$Fireplaces)
test$Yr.Sold <- as.factor(test$Yr.Sold)
test$MS.SubClass <- as.factor(test$MS.SubClass)
# convert to binary categories
test$Misc.Val <- as.factor(ifelse(test$Misc.Val == 0, 0, 1))
test$Pool.Area <- as.factor(ifelse(test$Pool.Area == 0, 0, 1))

numeric_cols <- sapply(test, is.numeric)
print(numeric_cols)

# handle missing values (fil them with zero)
test[is.na(test)] <- 0


# apply transformations
test$SalePrice <- log(test$SalePrice)
test$Lot.Area <- log(test$Lot.Area)
test$Lot.Frontage <- 1 / test$Lot.Frontage
test$BsmtFin.SF.1 <- log(test$BsmtFin.SF.1)
test$BsmtFin.SF.2 <- log(test$BsmtFin.SF.2)
test$Bsmt.Unf.SF <- sqrt(test$Bsmt.Unf.SF)
test$Total.Bsmt.SF <- log(test$Total.Bsmt.SF)
test$X1st.Flr.SF <- log(test$X1st.Flr.SF)
test$X2nd.Flr.SF <- log(test$X2nd.Flr.SF)
test$Gr.Liv.Area <- sqrt(test$Gr.Liv.Area)
test$Garage.Area <- log(test$Garage.Area)
test$Wood.Deck.SF <- log(test$Wood.Deck.SF)
test$Enclosed.Porch <- log(test$Enclosed.Porch)
test$X3Ssn.Porch <- 1 / test$X3Ssn.Porch
test$Screen.Porch <- log(test$Screen.Porch)


cols_to_check <- c("SalePrice", "Lot.Area", "BsmtFin.SF.1", "BsmtFin.SF.2", 
                   "Total.Bsmt.SF", "X1st.Flr.SF", "X2nd.Flr.SF", "Garage.Area", 
                   "Wood.Deck.SF", "Enclosed.Porch", "Screen.Porch")

# replace negative results with 0
for (col in cols_to_check) {
  test[[col]][test[[col]] < 0] <- 0
  test[[col]][!is.finite(test[[col]])]
}
inverse_cols <- c("Lot.Frontage", "X3Ssn.Porch")
for (col in inverse_cols) {
  test[[col]][!is.finite(test[[col]])] <- 0
}

test$Exter.Qual <- factor(test$Exter.Qual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
test$Kitchen.Qual <- factor(test$Kitchen.Qual, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)


# intraction terms

test$OverallQual_GrLivArea <- test$Overall.Qual * test$Gr.Liv.Area
test$ExterQual_OverallQual <- as.numeric(test$Exter.Qual) * test$Overall.Qual
test$GarageCars_GarageArea <- test$Garage.Cars * test$Garage.Area
test$TotalBsmtSF_GrLivArea <- test$Total.Bsmt.SF * test$Gr.Liv.Area
test$KitchenQual_OverallQual <- as.numeric(test$Kitchen.Qual) * test$Overall.Qual


numeric_cols <- setdiff(names(test)[sapply(test, is.numeric)], "SalePrice")

# standardize only numerical predictor columns by train data mean and std 

for (col in numeric_cols) {
  # Use training data's mean and std dev for standardization
  train_mean <- attr(data[[col]], "scaled:center")
  train_sd <- attr(data[[col]], "scaled:scale")
  # Standardize the column in the test data
  test[[col]] <- (test[[col]] - train_mean) / train_sd
}


data_features <- test[, !names(test) %in% "SalePrice"]
x1 <- sparse.model.matrix(~ ., data = data_features)[, -1]

train_columns <- colnames(x)
test_columns <- colnames(x1)
extra_in_test <- setdiff(test_columns, train_columns)
x1 <- x1[, !(colnames(x1) %in% extra_in_test)]
predicted_log_prices <- predict(final_model, x1 )

predicted_prices <- exp(predicted_log_prices)
# the new house predicted price
predicted_prices[1]