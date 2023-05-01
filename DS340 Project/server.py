from flask import Flask, render_template, request, redirect, url_for

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import LSTM_RNN_Code as LSTM

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])

def home():
    if(request.method == "POST" and request.form.get("home_button")):
        request.method = "GET"
        request.form = []
        
    request_meth = request.method
    print(request.form)
    if request.method == "POST":
        print("Home, new")
        print(request.method)
        print(request.form)
        
        stocknam = request.form["stock"]

        return redirect(url_for('s_name',stockname = stocknam))

    return render_template('webpage.html', request_method = request_meth)

@app.route("/stock/<string:stockname>", methods=["GET","POST"])
def s_name(stockname):
    print(request.method)
    if request.method == 'POST':
        if request.form["home_button"] == "Home":
            print(request.form)
            return redirect(url_for("home"))

    return render_template('display.html', stockname = stockname, prev_page=request.referrer)

@app.route('/plot.png/<stockname>')
def plot_png(stockname):
    
    print(stockname)
    fig = create_figure(stockname)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(stockname):
    """
    graph = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 3)]
    graph2 = [(0, 2), (0, 4), (1, 6), (2, 3), (2, 5), (3, 2)]
    fig = Figure()
    fig.plot(graph, color = 'black', label = 'Test')
    fig.plot(graph2, color = 'gold', label = 'Predicted')

    fig.legend()
    
    """
    fig = Figure(facecolor = 'black')
    axis = fig.add_subplot(1, 1, 1)
    
    test, predict = LSTM.lstm_function(stockname)
    
    y1 = test
    y2 = predict
    x1 = range(len(y1))
    x2 = range(len(y2))
    
    axis.plot(x1, y1, label='Line 1', color = 'white')
    axis.plot(x2, y2, label='Line 2', color = 'gold')
    
    axis.legend(['Test', 'Predicted'])
    axis.set_title(stockname, color = 'white')
    axis.tick_params(colors='white')
    axis.set_facecolor('black')


    return fig

if __name__ == '__main__':
    app.run(debug = True)


