#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

#import the data
data = pd.read_csv("Data Clean.csv")
image = Image.open("rumah.png")
st.title("Aplikasi Prediksi Harga Rumah")
st.image(image, use_column_width=True)

#checking the data
st.write("Aplikasi ini menggunakan Machine Learning untuk memperkirakan rentang harga rumah yang anda pilih. Ayo coba dan lihat hasilnya!")
check_data = st.checkbox("Tampilkan contoh data")
if check_data:
    st.write(data[1:10])
st.write("Sekarang mari kita cari tahu berapa harga rumah ketika kita memilih beberapa parameter")

#input the numbers
# Input untuk luas ruang tamu
sqft_liv = st.number_input("What is your square feet of living room?", 
                            min_value=int(data.sqft_living.min()), 
                            max_value=int(data.sqft_living.max()), 
                            value=int(data.sqft_living.mean()))

# Input untuk luas di atas
sqft_abo = st.number_input("What is your square feet of above?", 
                            min_value=int(data.sqft_above.min()), 
                            max_value=int(data.sqft_above.max()), 
                            value=int(data.sqft_above.mean()))

# Input untuk jumlah kamar mandi
bath = st.number_input("How many bathrooms?", 
                       min_value=int(data.bathrooms.min()), 
                       max_value=int(data.bathrooms.max()), 
                       value=int(data.bathrooms.mean()))

# Input untuk view
view = st.number_input("View?", 
                       min_value=int(data.view.min()), 
                       max_value=int(data.view.max()), 
                       value=int(data.view.mean()))

# Input untuk luas basement
sqft_bas = st.number_input("What is your square feet of basement?", 
                           min_value=int(data.sqft_basement.min()), 
                           max_value=int(data.sqft_basement.max()), 
                           value=int(data.sqft_basement.mean()))

# Input untuk kondisi
condition = st.number_input("Condition?", 
                            min_value=int(data.condition.min()), 
                            max_value=int(data.condition.max()), 
                            value=int(data.condition.mean()))

#splitting your data
X = data.drop('price', axis = 1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

#modelling step
#Linear Regression model
#import your model
model=LinearRegression()
#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[sqft_liv,sqft_abo,bath,view,sqft_bas,condition]])[0]
akurasi= np.sqrt(r2_score(y_test,model.predict(X_test)))

# =============================================================================
# #RandomForestModel
#model2 = RandomForestRegressor(random_state=0)
#model2.fit(X_train,y_train)
#model2.predict(X_test)
#errors = np.sqrt(mean_squared_error(y_test,model2.predict(X_test)))
#predictions = model2.predict([[sqft_liv,sqft_abo,bath,view,sqft_bas,condition]])[0]
#akurasi= np.sqrt(r2_score(y_test,model2.predict(X_test)))
# =============================================================================

# =============================================================================
# #DecissionTreeModel
#model3 = DecisionTreeRegressor(random_state= 45)
#model3.fit(X_train,y_train)
#model3.predict(X_test)
#errors = np.sqrt(mean_squared_error(y_test,model3.predict(X_test)))
#predictions = model3.predict([[sqft_liv,sqft_abo,bath,view,sqft_bas,condition]])[0]
#akurasi= np.sqrt(r2_score(y_test,model3.predict(X_test)))
# =============================================================================

# checking prediction house price
if st.button("Run me!", key="run_button"):
    # Menampilkan hasil prediksi harga rumah
    st.header("🏡 Your House Price Prediction")
    st.markdown(f"### **Estimated Price:** USD **{int(predictions):,}**")
    st.markdown(f"### **Price Range:** USD **{int(predictions - errors):,}** - USD **{int(predictions + errors):,}**")
    
    # Menampilkan akurasi
    st.markdown(f"### **Accuracy:** {akurasi:.2f}%")
