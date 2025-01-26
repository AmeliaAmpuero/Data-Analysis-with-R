#################0. Import required modules
# Install tidymodels
install.packages("rlang")
install.packages("tidymodels")
install.packages("tidyverse")
install.packages("RSQLite")
install.packages("readr")
install.packages("dplyr")
install.packages("rsample")
install.packages("glmnet")
install.packages("yardstick")
install.packages("stringr")

# Library for modeling
library(tidyverse)  # Loads ggplot2, dplyr, readr, etc.
library(tidymodels) # Loads recipes, parsnip, rsample, etc.
library(glmnet)      # Load glmnet for regularization models
library(yardstick)   # Load yardstick for performance metrics
library(stringr)     # Load stringr for string manipulation
library(Metrics)     # Load Metrics for additional evaluation metrics
#################1. Download NOAA Weather Dataset
# url where the data is located
url <- "https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz"

# download the file
download.file(url, destfile = "noaa-weather-sample-data.tar.gz")

# untar the file so we can get the csv only
# if you run this on your local machine, then can remove tar = "internal" 
untar("noaa-weather-sample-data.tar.gz")

#################2. Extract and Read into Project
# read_csv only 
jfk_weather <- read_csv("noaa-weather-sample-data/jfk_weather_sample.csv")

head(jfk_weather)

glimpse(jfk_weather)

#################3. Select Subset of Columns
sub_jfk_weather <- jfk_weather %>%
  select(HOURLYRelativeHumidity, HOURLYDRYBULBTEMPF, HOURLYPrecip, HOURLYWindSpeed, HOURLYStationPressure)

head(sub_jfk_weather,10)

#################4. Clean Up Columns
unique(sub_jfk_weather$HOURLYPrecip)

# Modify the HOURLYPrecip column
MOD_sub_jfk_weather <- sub_jfk_weather %>%
  mutate(HOURLYPrecip = ifelse(HOURLYPrecip == "T", "0.0", HOURLYPrecip), #Replace all the T values with "0.0"
         HOURLYPrecip = str_remove(HOURLYPrecip, pattern = "s$"))  # Remove "s" at the end
unique(MOD_sub_jfk_weather$HOURLYPrecip)
#################5. Convert Columns to Numerical Types
glimpse(MOD_sub_jfk_weather)
NUM_sub_jfk_weather <- MOD_sub_jfk_weather %>%
  mutate(HOURLYPrecip = as.numeric(HOURLYPrecip))
glimpse(NUM_sub_jfk_weather)

#################6.Rename Columns
weatherDF_jfk <- NUM_sub_jfk_weather %>% 
  replace_na(list(HOURLYRelativeHumidity=0, HOURLYDRYBULBTEMPF=0, HOURLYPrecip=0, HOURLYWindSpeed=0, HOURLYStationPressure=0)) %>% 
  rename("relative_humidity"="HOURLYRelativeHumidity", "dry_bulb_temp_f"="HOURLYDRYBULBTEMPF", "precip"="HOURLYPrecip", "wind_speed"="HOURLYWindSpeed", "station_pressure"="HOURLYStationPressure")

# Print the new dataframe to check the changes
print(weatherDF_jfk)

#################7. Exploratory Data Analysis
set.seed(1234)
weather_split <- initial_split(weatherDF_jfk,prop = 0.8)
train_data <- training(weather_split)
test_data <- testing(weather_split)

#Boxplots
ggplot(train_data, aes(y=relative_humidity))+geom_boxplot()
ggplot(train_data, aes(y=dry_bulb_temp_f))+geom_boxplot()
ggplot(train_data, aes(y=precip))+geom_boxplot()
ggplot(train_data, aes(y=wind_speed))+geom_boxplot()
ggplot(train_data, aes(y=station_pressure))+geom_boxplot()

#################8. Linear Regression
# Pick linear regression
lm_spec <- linear_reg() %>%
  # Set engine'
  set_engine(engine = "lm")
# Print the linear function
lm_spec

#precip ~ relative_humidity
train_fit1 <- lm_spec %>% 
  fit(precip ~ relative_humidity, data = train_data)
train_fit1
ggplot(train_data, aes(x=relative_humidity, y=precip))+geom_point()

#precip ~ dry_bulb_temp_f
train_fit2 <- lm_spec %>% 
  fit(precip ~ dry_bulb_temp_f, data = train_data)
train_fit2
ggplot(train_data, aes(x=dry_bulb_temp_f, y=precip))+geom_point()

#precip ~ wind_speed
train_fit3 <- lm_spec %>% 
  fit(precip ~ wind_speed, data = train_data)
train_fit3
ggplot(train_data, aes(x=wind_speed, y=precip))+geom_point()

#precip ~ station_pressure
train_fit4 <- lm_spec %>% 
  fit(precip ~ station_pressure, data = train_data)
train_fit4
ggplot(train_data, aes(x=station_pressure, y=precip))+geom_point()

#################9. Improve the Model
#1. MULTIPLE LINEAR REGRESSION
train_fit5 <- lm_spec %>% fit(precip ~ ., data = train_data)
train_fit5

train_results5 <- train_fit5 %>% 
  predict(new_data = train_data) %>% 
  mutate(truth=train_data$precip)
train_results5

#2. Ridge regularization
weather_recipe <- recipe(precip ~., data = train_data)
ridge_spec <- linear_reg(penalty = 0.1, mixture = 0) %>% set_engine("glmnet")
ridge_wf <- workflow() %>% add_recipe(weather_recipe)
ridge_fit <- ridge_wf %>% add_model(ridge_spec) %>% fit(data=train_data)
ridge_fit %>% pull_workflow_fit() %>% tidy()
train_result6 <- ridge_fit %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
head(train_result6)

#Grid Search 
tune_spec <- linear_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet")
weather_cvfolds <- vfold_cv(train_data)
lambda_grid <- grid_regular(levels = 50, penalty(range = c(-3, 0.3)))
ridge_grid <- tune_grid(ridge_wf %>% add_model(tune_spec), resamples=weather_cvfolds, grid=lambda_grid)

show_best(ridge_grid, metric="rmse")

#3.Elastic Net (L1 and L2) regularization
weather_recipe <- recipe(precip ~., data = train_data)
elastic_spec <- linear_reg(penalty = 0.00346, mixture = 0.2) %>% set_engine("glmnet")
elastic_wf <- workflow() %>% add_recipe(weather_recipe)
elastic_fit <- elastic_wf %>% add_model(elastic_spec) %>% fit(data=train_data)
elastic_fit %>% pull_workflow_fit() %>% tidy()

train_result7 <- elastic_fit %>% predict(new_data=train_data) %>% mutate(truth=train_data$precip)
head(train_result7)


#################10. Find the best model

train_results1 <- train_fit1 %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results1 <- train_fit1 %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results1, truth=truth, estimate=.pred)
rsq(test_results1, truth=truth, estimate=.pred)

train_results2 <- train_fit2 %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results2 <- train_fit2 %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results2, truth=truth, estimate=.pred)
rsq(test_results2, truth=truth, estimate=.pred)

train_results3 <- train_fit3 %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results3 <- train_fit3 %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results3, truth=truth, estimate=.pred)
rsq(test_results3, truth=truth, estimate=.pred)

train_results4 <- train_fit4 %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results4 <- train_fit4 %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results4, truth=truth, estimate=.pred)
rsq(test_results4, truth=truth, estimate=.pred)

train_results5 <- train_fit5 %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results5 <- train_fit5 %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results5, truth=truth, estimate=.pred)
rsq(test_results5, truth=truth, estimate=.pred)

train_results6 <- ridge_fit %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results6 <- ridge_fit %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results6, truth=truth, estimate=.pred)
rsq(test_results6, truth=truth, estimate=.pred)

train_results7 <- elastic_fit %>% predict(new_data = train_data) %>% mutate(truth=train_data$precip)
test_results7 <- elastic_fit %>% predict(new_data = test_data) %>% mutate(truth=test_data$precip)
rsq(train_results7, truth=truth, estimate=.pred)
rsq(test_results7, truth=truth, estimate=.pred)


model_names <- c("precip~humidity","precip~drybulbtemp","precip~windspeed","precip~stationpressure","Multiple Linear Regression", "Elastic Net Regularization")
train_error <- c(0.0203, 0.000336, 0.00263, 0.000162, 0.0302, 0.0285)
test_error <- c(0.0359, 0.000237, 0.0287, 0.000523, 0.0814, 0.0695)
comparison_df <- data.frame(model_names, train_error, test_error)
comparison_df

#CONCLUSION:
#The precip~drybulbtemp model has the lowest test error (0.000237) 
#and minimal difference between train and test errors, making it the most reliable and accurate among the options provided.