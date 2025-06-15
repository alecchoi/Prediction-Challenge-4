
# Load necessary libraries
library(caret)
library(ranger)

# Set the number of threads globally for ranger (here, 4 threads)
options(ranger.num.threads = 4)

# Set seed for reproducibility
set.seed(123)

# Read the training data from CSV file (ensure the file is in your working directory)
data <- read.csv("Prediction4Train.csv", stringsAsFactors = FALSE)

# Convert character columns to factors (including the target variable 'Deal')
data[] <- lapply(data, function(x) if(is.character(x)) factor(x) else x)
data$Deal <- factor(data$Deal)

# Split the data: 80% training and 20% testing
trainIndex <- createDataPartition(data$Deal, p = 0.80, list = FALSE)
train_data <- data[trainIndex, ]
test_data  <- data[-trainIndex, ]

# Set up training control with 5-fold cross-validation and up-sampling to balance classes
ctrl <- trainControl(method = "cv",
                     number = 5,
                     sampling = "up",
                     classProbs = TRUE)

# Train a ranger model using caret with tuning; num.threads is explicitly set to 4
set.seed(123)
tuned_model <- train(Deal ~ ., 
                     data = train_data,
                     method = "ranger",
                     trControl = ctrl,
                     tuneLength = 3,
                     importance = "impurity",
                     num.threads = 4)

# Print the tuned model parameters
print(tuned_model)

# Make predictions on the test set using 4 threads
predictions <- predict(tuned_model, newdata = test_data, num.threads = 4)

# Evaluate the model using a confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$Deal)
print(conf_matrix)

# Calculate and print the overall accuracy percentage
accuracy <- conf_matrix$overall["Accuracy"] * 100
cat("Accuracy:", round(accuracy, 2), "%\n")


# ---------------------------
# PART 2: Predicting on Students' Test Data and Creating Submission File
# ---------------------------

# Read the students' test data (ensure the file "Prediction4StudentsTest.csv" is in your working directory)
test_students <- read.csv("Prediction4StudentsTest.csv", stringsAsFactors = FALSE)

# For each predictor in the training data that is a factor, update the corresponding column in test_students
for(col in names(train_data)){
  if(is.factor(train_data[[col]]) && col %in% names(test_students)){
    # Force test_students column to have the same levels as in training data.
    test_students[[col]] <- factor(test_students[[col]], levels = levels(train_data[[col]]))
    
    # Identify any NA (resulting from unknown levels) and replace with the mode from training data.
    mode_val <- names(sort(table(train_data[[col]]), decreasing = TRUE))[1]
    test_students[[col]][is.na(test_students[[col]])] <- mode_val
  }
}

# Generate predictions on the students' test data using the tuned model with 4 threads
students_predictions <- predict(tuned_model, newdata = test_students, num.threads = 4)

# Create a submission data frame with 'ID' and predicted 'Deal'
submission <- data.frame(ID = test_students$ID, Deal = students_predictions)

# Write the submission file to CSV (without row names)
write.csv(submission, "submission.csv", row.names = FALSE)
cat("Submission file 'submission.csv' created.\n")
