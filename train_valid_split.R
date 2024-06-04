# Author : Claudéric DeRoy
# Goal : split the HCP dataset for between a 80% subdataset for machine learning
#        training and a 20% for independent testing
# Last date of modification : 06/06/2024

library(tidyverse)

data <- read.csv("/home/clauderic/Maîtrise Psychologie/Xanthy's project/Final_HCP_database.csv", header = TRUE)

# creating the 20% independent test with a 50% partition between man and woman
valid_size <- floor(nrow(data)*0.20)
valid_set <- data.frame()

man <- data[data$Gender == 1,]
woman <- data[data$Gender == 0,]

index_man <- sample(1:nrow(man), (valid_size/2))
index_woman <- sample(1:nrow(woman), (valid_size/2))

# adding man to validation set
for(i in index_man){
  indexing <- man[i,]$index
  valid_set <- rbind(valid_set, data[data$index == indexing,])
  data <- data[data$index != indexing,]
}

# adding woman to validation set
for(i in index_woman){
  indexing <- woman[i,]$index
  valid_set <- rbind(valid_set, data[data$index == indexing,])
  data <- data[data$index != indexing,]
}

# reordering the validation set
valid_set <- valid_set[order(valid_set$index),]

# save the valid set and the left over set
write.csv(valid_set, "/home/clauderic/Maîtrise Psychologie/Xanthy's project/validation_set.csv", row.names = TRUE)
write.csv(data, "/home/clauderic/Maîtrise Psychologie/Xanthy's project/testing_set.csv", row.names = TRUE)