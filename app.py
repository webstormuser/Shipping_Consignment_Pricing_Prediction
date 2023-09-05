from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/prediction/input')
def input_prediction():
    return render_template('input.html')

@app.route('/prediction/batch')
def batch_prediction():
    return render_template('batch_prediction.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(port=5002,debug=True)
