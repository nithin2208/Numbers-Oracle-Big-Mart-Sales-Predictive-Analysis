from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model
model = pickle.load(open(r"C:\Users\esamp\OneDrive\Desktop\BigMartSalesPrediction\model (1).pkl", 'rb'))

# Initialize the app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Weight = float(request.form['Weight'])
        ProductVisibility = float(request.form['ProductVisibility'])
        ProductType = float(request.form['ProductType'])
        MRP = float(request.form['MRP'])
        Outlet_Years = float(request.form['Outlet_Years'])
        Outlet = float(request.form['Outlet'])
        FatContent = float(request.form['FatContent'])
        OutletSize = float(request.form['OutletSize'])
        OutletType = float(request.form['OutletType'])
        New_Item_Type = float(request.form['New_Item_Type'])

        # Prepare the input for prediction
        X = np.array([[Weight, ProductVisibility, ProductType, MRP, Outlet_Years, Outlet, FatContent, OutletSize, OutletType, New_Item_Type]])
        
        # Predict the output
        output = model.predict(X)[0]

        return render_template('result.html', output=output)
    return render_template("predict.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
