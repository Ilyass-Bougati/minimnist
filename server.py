from flask import Flask, request, render_template
import torch
import torch.nn.functional as f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth", weights_only=False)
model.eval()
app = Flask(__name__)

@app.route("/ai", methods=["POST"])
def ai():
    content = request.json
    img = content.get("image")
    if img == None:
        return "err"
    else:
        res = []
        for i in range(12):
            for j in range(6):
                res.append(img[i][j])

        image_tensor = torch.tensor(res).float().flatten()
        print(image_tensor)
        res = torch.softmax(model(image_tensor), dim=0)

        return {
            "res": [float(i) for i in res]
        }


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print()
    app.run(port = 5000, debug=True)