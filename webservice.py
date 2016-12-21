from flask import Flask, request, jsonify
import string
import argparse
import learn_news


app = Flask(__name__)


def clean(phrase):
    cleaned = ''
    for char in phrase:
        if char in string.ascii_letters + ' ':
            cleaned += char
    return cleaned


@app.route('/api/v1/explain')
def explain():
    phrase = request.args.get('phrase')
    phrase = clean(phrase)
    print('phrase', phrase)
    # return 'ok', 200
    res = model.query(phrase)
    print('res', res)
    return jsonify({'res': res})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', type=str, default='nb', help='[nb|sgd|rbf]')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--port', type=int, default=15000)
    args = parser.parse_args()

    # learn_news.train_model(args.trainer)
    model = learn_news.Model(args.trainer)
    model.train()

    app.run(debug=args.debug, port=args.port)
