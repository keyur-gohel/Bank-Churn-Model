# ANN (Artificial Neural Network)

# Importing Dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoing Categorical Variable as Factor
dataset$Geography = as.numeric(factor(dataset$Geography,
                           levels = c('France', 'Spain', 'Germany'),
                           labels = c(1, 2, 3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                                      levels = c('Female', 'Male'),
                                      labels = c(1, 2)))

# Spliting the Datset into Training and Test Set
library(caTools)
set.seed(100)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_Set = subset(dataset, split == TRUE)
test_Set = subset(dataset, split == FALSE)

# Feature Scalling
training_Set[-11] = scale(training_Set[-11])
test_Set[-11] = scale(test_Set[-11])

# Fitting ANN to Training Set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited', 
                              training_frame = as.h2o(training_Set),
                              activation = 'Rectifier',
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)


# Predicting Values
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_Set[-11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Confusion matrix
cm = table(test_Set[, 11], y_pred)

h2o.shutdown()
