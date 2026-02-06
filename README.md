# Food-Hub-Python-Project

Context

The number of restaurants in New York is increasing day by day. Lots of students and busy professionals rely on those restaurants due to their hectic lifestyles. Online food delivery service is a great option for them. It provides them with good food from their favorite restaurants. A food aggregator company FoodHub offers access to multiple restaurants through a single smartphone app.

The app allows the restaurants to receive a direct online order from a customer. The app assigns a delivery person from the company to pick up the order after it is confirmed by the restaurant. The delivery person then uses the map to reach the restaurant and waits for the food package. Once the food package is handed over to the delivery person, he/she confirms the pick-up in the app and travels to the customer's location to deliver the food. The delivery person confirms the drop-off in the app after delivering the food package to the customer. The customer can rate the order in the app. The food aggregator earns money by collecting a fixed margin of the delivery order from the restaurants.

Objective

The food aggregator company has stored the data of the different orders made by the registered customers in their online portal. They want to analyze the data to get a fair idea about the demand of different restaurants which will help them in enhancing their customer experience. Suppose you are hired as a Data Scientist in this company and the Data Science team has shared some of the key questions that need to be answered. Perform the data analysis to find answers to these questions that will help the company to improve the business.

Data Description

The data contains the different data related to a food order. The detailed data dictionary is given below.

Data Dictionary

order_id: Unique ID of the order
customer_id: ID of the customer who ordered the food
restaurant_name: Name of the restaurant
cuisine_type: Cuisine ordered by the customer
cost: Cost of the order
day_of_the_week: Indicates whether the order is placed on a weekday or weekend (The weekday is from Monday to Friday and the weekend is Saturday and Sunday)
rating: Rating given by the customer out of 5
food_preparation_time: Time (in minutes) taken by the restaurant to prepare the food. This is calculated by taking the difference between the timestamps of the restaurant's order confirmation and the delivery person's pick-up confirmation.
delivery_time: Time (in minutes) taken by the delivery person to deliver the food package. This is calculated by taking the difference between the timestamps of the delivery person's pick-up confirmation and drop-off information
Let us start by importing the required libraries
# import libraries for data manipulation
import numpy as np
import pandas as pd

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
Understanding the structure of the data
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
# read the data
df = pd.read_csv('/content/drive/MyDrive/foodhub_order.csv')
# returns the first 5 rows
df.head()
order_id	customer_id	restaurant_name	cuisine_type	cost_of_the_order	day_of_the_week	rating	food_preparation_time	delivery_time
0	1477147	337525	Hangawi	Korean	30.75	Weekend	Not given	25	20
1	1477685	358141	Blue Ribbon Sushi Izakaya	Japanese	12.08	Weekend	Not given	25	23
2	1477070	66393	Cafe Habana	Mexican	12.23	Weekday	5	23	28
3	1477334	106968	Blue Ribbon Fried Chicken	American	29.20	Weekend	3	25	15
4	1478249	76942	Dirty Bird to Go	American	11.59	Weekday	4	25	24
Observations:

The DataFrame has 9 columns as mentioned in the Data Dictionary. Data in each row corresponds to the order placed by a customer.

Question 1: How many rows and columns are present in the data? [0.5 mark]
num_rows, num_columns = df.shape

print("number of rows:", num_rows)
print("number of columns:", num_columns)
number of rows: 1898
number of columns: 9
Observations:
Number of Rows: 1898, Number of Columns: 9

Question 2: What are the datatypes of the different columns in the dataset? (The info() function can be used) [0.5 mark]
# Use info() to print a concise summary of the DataFrame
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1898 entries, 0 to 1897
Data columns (total 9 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   order_id               1898 non-null   int64  
 1   customer_id            1898 non-null   int64  
 2   restaurant_name        1898 non-null   object 
 3   cuisine_type           1898 non-null   object 
 4   cost_of_the_order      1898 non-null   float64
 5   day_of_the_week        1898 non-null   object 
 6   rating                 1898 non-null   object 
 7   food_preparation_time  1898 non-null   int64  
 8   delivery_time          1898 non-null   int64  
dtypes: float64(1), int64(4), object(4)
memory usage: 133.6+ KB
Observations:
We have 1898 entries for this dataset. 9 columns broken down with below Dtypes:

(1) Float64 (4) int64 (4) object

Question 3: Are there any missing values in the data? If yes, treat them using an appropriate method. [1 mark]
total_missing_values = df.isna().sum()
# check to see if there are any missing values
print("Total number of missing values:", total_missing_values)
Total number of missing values: order_id                 0
customer_id              0
restaurant_name          0
cuisine_type             0
cost_of_the_order        0
day_of_the_week          0
rating                   0
food_preparation_time    0
delivery_time            0
dtype: int64
Observations:
There are no missing values for this dataset.

Question 4: Check the statistical summary of the data. What is the minimum, average, and maximum time it takes for food to be prepared once an order is placed? [2 marks]
# run statistical summary of data via describe fx
df = df.describe()
print(df)
           order_id    customer_id  cost_of_the_order  food_preparation_time  \
count  1.898000e+03    1898.000000        1898.000000            1898.000000   
mean   1.477496e+06  171168.478398          16.498851              27.371970   
std    5.480497e+02  113698.139743           7.483812               4.632481   
min    1.476547e+06    1311.000000           4.470000              20.000000   
25%    1.477021e+06   77787.750000          12.080000              23.000000   
50%    1.477496e+06  128600.000000          14.140000              27.000000   
75%    1.477970e+06  270525.000000          22.297500              31.000000   
max    1.478444e+06  405334.000000          35.410000              35.000000   

       delivery_time  
count    1898.000000  
mean       24.161749  
std         4.972637  
min        15.000000  
25%        20.000000  
50%        25.000000  
75%        28.000000  
max        33.000000  
Observations:
Food preparation times from an order being placed:

Min = 20 mins Max = 35 Mins Average = 27.37

Question 5: How many orders are not rated? [1 mark]
# use total_ratings to identify rating values
total_ratings = df['rating'].value_counts()
print(total_ratings)
rating
Not given    736
5            588
4            386
3            188
Name: count, dtype: int64
Observations:
We have a total of 736 ratings not given.

Exploratory Data Analysis (EDA)
Univariate Analysis
Question 6: Explore all the variables and provide observations on their distributions. (Generally, histograms, boxplots, countplots, etc. are used for univariate exploration.) [9 marks]
# breaking down the columns by data type to outline univariate and bivariate analysis variables
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)
Numerical columns: Index(['order_id', 'customer_id', 'cost_of_the_order', 'food_preparation_time',
       'delivery_time'],
      dtype='object')
Categorical columns: Index(['restaurant_name', 'cuisine_type', 'day_of_the_week', 'rating'], dtype='object')
Observations: There are 5 numerical columns identified as order ID, customer ID, Cost of the Order, Food Preparation Time and Delivery Time. There are 4 categorical columns identified as Restaurant Name, Cuisine Type, Day of the Week and Rating.
Numerical Variables
Cost of the Order
# visual representation of the cost of orders
sns.histplot(data=df, x='cost_of_the_order', kde=True)
plt.show()
sns.boxplot(data=df, x='cost_of_the_order')
plt.show()


print(df['cost_of_the_order'].describe())
count    1898.000000
mean       16.498851
std         7.483812
min         4.470000
25%        12.080000
50%        14.140000
75%        22.297500
max        35.410000
Name: cost_of_the_order, dtype: float64
Food Preparation Time
sns.histplot(data=df, x='food_preparation_time', kde=True)
plt.show()
sns.boxplot(data=df, x='food_preparation_time')
plt.show()


print(df['food_preparation_time'].describe())
count    1898.000000
mean       27.371970
std         4.632481
min        20.000000
25%        23.000000
50%        27.000000
75%        31.000000
max        35.000000
Name: food_preparation_time, dtype: float64
Delivery Time
sns.histplot(data=df, x='delivery_time', kde=True)
plt.show()
sns.boxplot(data=df, x='delivery_time')
plt.show()


print(df['delivery_time'].describe())
count    1898.000000
mean       24.161749
std         4.972637
min        15.000000
25%        20.000000
50%        25.000000
75%        28.000000
max        33.000000
Name: delivery_time, dtype: float64
Average Cost of Order is: 16 Dollars. Fastest Preparation time is: 20 minutes. Fastest Delivery time is: 15 minutes. Suggestion would be finding a way to incentivize the cooks/restaurants to get orders prepared more quickly as most delivery times are managable and by comparison prep time is slow. ulitmately this is leading to less frequent orders at an average of $16.

Categorical Variables
Cuisine Type
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.countplot(y=df['cuisine_type'], order=df['cuisine_type'].value_counts().index)
plt.title('Countplot of Cuisine Type')

print(df['cuisine_type'].value_counts())
cuisine_type
American          584
Japanese          470
Italian           298
Chinese           215
Mexican            77
Indian             73
Middle Eastern     49
Mediterranean      46
Thai               19
French             18
Southern           17
Korean             13
Spanish            12
Vietnamese          7
Name: count, dtype: int64

Day of the Week
plt.subplot(1, 3, 2)
sns.countplot(y=df['day_of_the_week'], order=df['day_of_the_week'].value_counts().index)
plt.title('Countplot of Day of the Week')

print(df['day_of_the_week'].value_counts())
day_of_the_week
Weekend    1351
Weekday     547
Name: count, dtype: int64

Rating
plt.subplot(1, 3, 3)
sns.countplot(y=df['rating'], order=df['rating'].value_counts().index)
plt.title('Countplot of Rating')

plt.tight_layout()
plt.show()

print("Summary statistics for 'rating':")
print(df['rating'].value_counts())

Summary statistics for 'rating':
rating
Not given    736
5            588
4            386
3            188
Name: count, dtype: int64
Categorical observations:

Need to increase orders with restaurants not categorzied as american, japanese, chinese and italian.
See a 100% increase of orders on the weekends.
Significant amount of orders received without any rating.
Question 7: Which are the top 5 restaurants in terms of the number of orders received? [1 mark]
# group the data by restaurant, count orders from each one
restaurant_orders = df.groupby('restaurant_name')['order_id'].count()
top_5_restaurants = restaurant_orders_sorted.head(5)
# display top 5 restaurants
print("Top 5 Restaurants by Order Count:")
print(top_5_restaurants)
Top 5 Restaurants by Order Count:
restaurant_name
Shake Shack                  219
The Meatball Shop            132
Blue Ribbon Sushi            119
Blue Ribbon Fried Chicken     96
Parm                          68
Name: order_id, dtype: int64
Observations: the top 5 restaurants, in order of total orders received are Shake Shack, The Meatball Shop, Blue Ribbon Sushi, Blue Ribbon Fried Chicken, Parm. Shake Shack received the most orders and Parm rounded out the top 5.
Question 8: Which is the most popular cuisine on weekends? [1 mark]
# display table identifying the top cuisine option on weekends
contingency_table = pd.crosstab(df['cuisine_type'], df['day_of_the_week'])
print(contingency_table)
day_of_the_week  Weekday  Weekend
cuisine_type                     
American             169      415
Chinese               52      163
French                 5       13
Indian                24       49
Italian               91      207
Japanese             135      335
Korean                 2       11
Mediterranean         14       32
Mexican               24       53
Middle Eastern        17       32
Southern               6       11
Spanish                1       11
Thai                   4       15
Vietnamese             3        4
Observations: The Top Choice for Cuisine on the Weekends is American.
Question 9: What percentage of the orders cost more than 20 dollars? [2 marks]
#filter for orders > $20, calculate the number of orders, calculate the total
orders_over_20 = df[df['cost_of_the_order'] > 20]
num_orders_over_20 = orders_over_20.shape[0]
total_orders = df.shape[0]
# display percentage
percentage_over_20 = (num_orders_over_20 / total_orders) * 100
print("Percentage of orders that cost more than $20: {:.2f}%".format(percentage_over_20))
Percentage of orders that cost more than $20: 29.24%
Observations: 29.24% of the orders are over $20.00.
Question 10: What is the mean order delivery time? [1 mark]
#calculate the mean of the delivery time
delivery_time_mean = df['delivery_time'].mean()
print("The mean delivery time is:", delivery_time_mean)
The mean delivery time is: 24.161749209694417
Observations: the mean of the delivery time is about 24 minutes.
Question 11: The company has decided to give 20% discount vouchers to the top 5 most frequent customers. Find the IDs of these customers and the number of orders they placed. [1 mark]
# group by customer id and count orders from each customer
customer_order_counts = df['customer_id'].value_counts()
#isolate the top 5
top_5_customers = customer_order_counts.head(5)

print("Top 5 most frequent customers and the number of orders they placed:")
print(top_5_customers)
Top 5 most frequent customers and the number of orders they placed:
customer_id
52832     13
47440     10
83287      9
250494     8
259341     7
Name: count, dtype: int64
Observations: Customer 52832 has placed the most orders with 13, followed by Customer 47440 (10 orders), Customer 83287 (9 orders), Customer 250494 (8 orders) and lastly Customer 259341 (7 orders). Those are the top 5 customers most frequently ordering customers.
Multivariate Analysis
Question 12: Perform a multivariate analysis to explore relationships between the important variables in the dataset. (It is a good idea to explore relations between numerical variables as well as relations between numerical and categorical variables) [10 marks]
Bi/Multivariate Analysis'
Numerical/Numerical Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(x='cost_of_the_order', y='delivery_time', data=df)
plt.title('Scatter Plot of Cost vs. Delivery Time')
plt.xlabel('Order Cost')
plt.ylabel('Delivery Time')
plt.show()

correlation = df['cost_of_the_order'].corr(df['delivery_time'])
print(f"Correlation between cost and delivery time: {correlation}")

Correlation between cost and delivery time: -0.029949184900648816
Numerical/Numerical Observations: There are a concerning amount of expensive orders arriving to customers significantly later than our mean delivery time resulting in a negative correlation.

Numerical/Categorical Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='cuisine_type', y='cost_of_the_order', data=df)
plt.title('Box Plot of Cost by Cuisine Type')
plt.xlabel('Cuisine Type')
plt.ylabel('Order Cost')
plt.xticks(rotation=90)
plt.show()

Numerical/Categorical Observations: Southern cuisine seems to have the greatest amount of flux in cost vs cuisine type. Korean has the least.

Categorical/Categorical Analysis
plt.figure(figsize=(12, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Cuisine Type by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Cuisine Type')
plt.show()

contingency_table = pd.crosstab(df['cuisine_type'], df['day_of_the_week'])
print(contingency_table)

day_of_the_week  Weekday  Weekend
cuisine_type                     
American             169      415
Chinese               52      163
French                 5       13
Indian                24       49
Italian               91      207
Japanese             135      335
Korean                 2       11
Mediterranean         14       32
Mexican               24       53
Middle Eastern        17       32
Southern               6       11
Spanish                1       11
Thai                   4       15
Vietnamese             3        4
Categorical/Categorical Observations: American, Italian and Japanese cuisines see a greater than 100% increase from weekday to weekend orders. Every cuisine type sees and overall increase by the time the weekend arrives.

Question 13: The company wants to provide a promotional offer in the advertisement of the restaurants. The condition to get the offer is that the restaurants must have a rating count of more than 50 and the average rating should be greater than 4. Find the restaurants fulfilling the criteria to get the promotional offer. [3 marks]
# identify the unique values
print(df['rating'].unique())
['Not given' '5' '3' '4']
# remove the not given values
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# group by restaurant with a count of rating and average
rating_group = df.groupby('restaurant_name').agg(
    rating_count=('rating', 'size'),
    avg_rating=('rating', 'mean')
)

#identify the qualified restaurants
qualified_restaurants = rating_group[(rating_group['rating_count'] > 50) & (rating_group['avg_rating'] > 4)]

qualified_restaurants = qualified_restaurants.sort_values(by='avg_rating', ascending=False)

print("Restaurants qualfied for the promotional offer:")
print(qualified_restaurants)
Restaurants qualfied for the promotional offer:
                           rating_count  avg_rating
restaurant_name                                    
The Meatball Shop                   132    4.511905
Blue Ribbon Fried Chicken            96    4.328125
Shake Shack                         219    4.278195
RedFarm Broadway                     59    4.243902
Blue Ribbon Sushi                   119    4.219178
RedFarm Hudson                       55    4.176471
Parm                                 68    4.128205
Observations: We have identified 7 restaurants that qualify for the promotion, having a Customer Rating over 4. Those restaurants are The Meatball Shop, Blue Ribbon, Fried Chicken, Shake Shack, RedFarm Broadway, Blue Ribbon Sushi, RedFarm Hudson and Parm.

Question 14: The company charges the restaurant 25% on the orders having cost greater than 20 dollars and 15% on the orders having cost greater than 5 dollars. Find the net revenue generated by the company across all orders. [3 marks]
#define new revenue column
df['revenue'] = 0
# use the loc operation to identify the two key pieces of data for surcharges
df.loc[df['cost_of_the_order'] > 20, 'revenue'] = df['cost_of_the_order'] * 0.25
df.loc[(df['cost_of_the_order'] <= 20) & (df['cost_of_the_order'] > 5), 'revenue'] = df['cost_of_the_order'] * 0.15

net_revenue = df['revenue'].sum()

print("Net revenue w/ surcharge:", net_revenue)
Net revenue w/ surcharge: 6166.303
Observations: The Net Revenue Generated is $6,166.30.
Question 15: The company wants to analyze the total time required to deliver the food. What percentage of orders take more than 60 minutes to get delivered from the time the order is placed? (The food has to be prepared and then delivered.) [2 marks]
# calculate the total time needed for the orders
df['total_time'] = df['food_preparation_time'] + df['delivery_time']
# identify the orders that take longer than 60 mins
num_orders_over_60 = df[df['total_time'] > 60].shape[0]
total_orders = df.shape[0]
percentage_over_60 = (num_orders_over_60 / total_orders) * 100

print("% Orders over 60 min total time:", percentage_over_60)
% Orders over 60 min total time: 10.537407797681771
Observations: The Percentage of Orders that take over 60 minutes from the time the order is placed to delivery is about 11% of our orders.
Question 16: The company wants to analyze the delivery time of the orders on weekdays and weekends. How does the mean delivery time vary during weekdays and weekends? [2 marks]
print(df['day_of_the_week'].unique())
['Weekend' 'Weekday']
# convert to lowercase
df['day_of_the_week'] = df['day_of_the_week'].str.lower()
#calculate the time for weekdays and weekends
mean_delivery_time_weekdays = df[df['day_of_the_week'] == 'weekday']['delivery_time'].mean()
mean_delivery_time_weekends = df[df['day_of_the_week'] == 'weekend']['delivery_time'].mean()

print("Mean delivery time for weekdays:", mean_delivery_time_weekdays)
print("Mean delivery time for weekends:", mean_delivery_time_weekends)
Mean delivery time during weekdays: 28.340036563071298
Mean delivery time during weekends: 22.4700222057735
Observations: The mean delivery time on weekdays is about 28 minutes and the mean delivery time on weekends is about 22 minutes.
Conclusion and Recommendations
Question 17: What are your conclusions from the analysis? What recommendations would you like to share to help improve the business? (You can use cuisine type and feedback ratings to drive your business recommendations.) [6 marks]
Conclusions:

The amount of time it takes more expensive orders to arrive customers is concerning. Preparation stage would be understandable as they are large orders but the uptick in delivery time should be handled more carefully. Large orders are highly valuable customers.
Room for the sales between week and weekend to sinch up.
There is a need to drive up sales during the week with less popular cuisine options.
Need to expedite the process between the time the order is placed and the time the order arrives to the customer, more specifically in the preparation stage.
Orders on the weekend are delivered more quickly than they are during the week when business shows to be slower.
Recommendations:

Expand the amount of customers you are incentivizing for return orders. Epsecially with weekday deals.
Examine reducing the surcharge associated with orders for restaurants with cuisine types that are less popular.
The time it takes to deliver on weekdays needs to be expedited, if customers are ordering less frequenetly during the week, they should be able to expect service to be quicker while times are slow.
Incentivize customers by rating their experience for more data on areas of improvement. There is an overwhelming amount of ratings missed currently.
Work with restaurants to see if anything could be improved on our side to help expedite the preparation time, this seems to be a weakness as well.
