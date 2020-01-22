library(imager)
library(dplyr)
library(png)
library(jpeg)
library(EBImage)
library(keras)
library(tensorflow)
library(evaluate)


setwd("/Volumes/Extern1/Bulk_Images/NAIP/sliced_images")

files = list.files()

images <- list()
#images[[1]] <- resize(readImage(files[1]),30,30)
pb<-txtProgressBar(min = 1, max = 1764, style = 3)
for (i in 1:length(files)){
  images[[i]] <- resize(readImage(files[i]),32,32)
  images[[1764+i]] <- resize(flop(readImage(files[i])),32,32)
  images[[3528+i]] <- resize(flip(readImage(files[i])),32,32)
  setTxtProgressBar(pb, i)
}


str(images)
plot(images[[20]])

#image_list <- combine(images)
str(image_list)

image_list <- list()
image_list <- combine(images[1:250])
image_list1 <- combine(images[251:500])
image_list <- combine(image_list,image_list1)
image_list1 <- combine(images[501:750])
image_list <- combine(image_list,image_list1)
image_list1 <- combine(images[751:1000])
image_list <- combine(image_list,image_list1)
image_list1 <- combine(images[1001:1250])
image_list <- combine(image_list,image_list1)
image_list1 <- combine(images[1251:1500])
image_list <-combine(image_list,image_list1)
image_list1 <- combine(images[1501:1750])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[1751:2000])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[2001:2250])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[2251:2500])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[2501:2750])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[2751:3000])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[3001:3200])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[3201:3500])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[3501:3750])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[3751:4000])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[4001:4250])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[4251:4500])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[4501:4750])
image_list <- combine(image_list, image_list1)
image_list1 <- combine(images[4751:5000])
image_list <- combine(image_list, image_list1)
image_list_test <- combine(images[5001:5292])

#image_list_test <- combine(images[1501:1764])
#image_list_train <- combine(image_list)

#str(image_list_train)
image_list <- aperm(image_list,c(4,1,2,3))
#image_list_train <- aperm(image_list, c(4,1,2,3))
image_list_test <-aperm(image_list_test, c(4,1,2,3))
str(image_list)

land_class <- read.csv("~/Desktop/Image_Classification.csv")
land_class_train <- land_class[1:5000,]
land_class_test <- land_class[5001:5292,]

land_class_train <- as.numeric(land_class_train)
land_class_test <- as.numeric(land_class_test)
#land_class <- as.numeric(land_class$Land_Type)
#land_class <- land_class[1:1764]

land_labels_train <- to_categorical(land_class_train)
land_labels_test <- to_categorical(land_class_test)
#land_labels <- to_categorical(land_class)

binary_class_train <- ifelse(land_class_train == 6,1,0)
binary_class_test <- ifelse(land_class_test == 6,1,0)

binary_labels_train <- to_categorical(binary_class_train)
binary_labels_test <- to_categorical(binary_class_test)

CNN_first <- keras_model_sequential()


trial <- CNN_first %>%
  layer_conv_2d(filters = 32, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(32,32,3))%>%
  layer_conv_2d(filters = 32, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>% #Pooling Layer
  layer_dropout(rate = 0.25) %>% #Dropout Layer
  layer_conv_2d(filters = 64, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu')%>%
  layer_conv_2d(filters = 64, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2))%>% #Pooling Layer
  layer_dropout(rate = 0.25)%>% #Dropout Layer
  layer_flatten()%>% #Flattening Layer
  layer_dense(units = 256, activation = 'relu')%>% #Connected Neural Net
  layer_dropout(rate = 0.25)%>% #Dropout Layer
  layer_dense(units = 8, activation = 'softmax')%>% #Neural Net
  compile( loss = 'categorical_crossentropy', #Output Layer
           optimizer = optimizer_sgd(lr = 0.01,
                                     decay = 1e-6,
                                     momentum = 0.9,
                                     nesterov = T),
           metrics = c('accuracy'))

summary(trial)
summary(CNN_first)

CNN_sec <- keras_model_sequential()
trial2 <- CNN_sec %>%
  layer_conv_2d(filters = 32, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(32,32,3))%>%
  layer_conv_2d(filters = 32, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>% #Pooling Layer
  layer_dropout(rate = 0.25) %>% #Dropout Layer
  layer_conv_2d(filters = 64, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu')%>%
  layer_conv_2d(filters = 64, #Convolutional Layer
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2))%>% #Pooling Layer
  layer_dropout(rate = 0.25)%>% #Dropout Layer
  layer_flatten()%>% #Flattening Layer
  layer_dense(units = 256, activation = 'relu')%>% #Connected Neural Net
  layer_dropout(rate = 0.25)%>% #Dropout Layer
  layer_dense(units = 2, activation = 'softmax')%>% #Neural Net
  compile( loss = 'binary_crossentropy', #Output Layer
           optimizer = optimizer_sgd(lr = 0.01,
                                     decay = 1e-6,
                                     momentum = 0.9,
                                     nesterov = T),
           metrics = c('accuracy'))

#Fit Model
dim(image_list_test)
image_list <- aperm(image_list,c(4,1,2,3))
image_list_test <- aperm(image_list_test,c(2,3,4,1))

history <- trial%>%
  fit(image_list,
      land_labels_train,
      epochs = 250,
      batch_size = 32,
      validation_split = 0.125)
plot(history)

binary_c <- trial2%>%
  fit(image_list,
      binary_labels_train,
      epochs = 250,
      batch_size = 32,
      validation_split = 0.125)
plot(binary_c)

trial%>% evaluate(image_list, land_labels_train)
pred <- trial%>%predict_classes(image_list)
table(Predicted = pred, Actual = land_class_train)

prob <- trial %>%
  predict_proba(image_list)
cbind(prob, Predicted_class = pred, Actual = land_class_train)

pred_test <- trial%>%predict_classes(image_list_test)
table(Predicted = pred_test, Actual = land_class_test)

trial2 %>%evaluate(image_list, binary_labels_train)
bin_pred <- trial2%>%predict_classes(image_list)
table(Predicted = bin_pred, Actual = binary_class_train)

prob2 <- trial2 %>%
  predict_proba(image_list)
cbind(prob2, Predicted_class = bin_pred, Actual = land_class_train)

pred_test_bin <- trial2%>%predict_classes(image_list_test)
table(Predicted = pred_test_bin, Actual = binary_class_test)

