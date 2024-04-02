from sklearn.linear_model import LinearRegression
import numpy as np

square_feet = np.array([1000, 1500, 1200, 1700, 800])
bedrooms = np.array([2, 3, 2, 4, 1])
y = np.array([300000, 450000, 350000, 550000, 250000])

model = LinearRegression()
X = np.vstack((square_feet, bedrooms)).T
model.fit(X, y)

new_data = np.array([[950, 3]])
predicted_price = model.predict(new_data)

print(f"Предсказанная цена недвижимости: {predicted_price[0]:.2f}")