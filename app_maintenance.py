from flask import Flask, render_template, request, jsonify, send_from_directory
import json

app = Flask(__name__)



# ウェブページを表示するルート
@app.route('/')
def index():
    return render_template('index_maintenance.html')
   

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=11111)
    # app.run(debug=True)
