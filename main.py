import os
import time

from bottle import Bottle
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
from FeatureCloud.app.engine.app import app, State as OpState
from utils.utils import is_native, is_standalone
import states

server = Bottle()

STANDALONE_POLL_INTERVAL = 0.5


def run_app():
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)


def run_standalone():
    """Start the state machine without the FeatureCloud controller and exit when done."""
    app.register()
    app.handle_setup(client_id='1', coordinator=True, clients=['1'])

    while True:
        current = app.current_state
        if current is not None and current.name == 'terminal':
            os._exit(0)
        if app.status_finished and app.status_state == OpState.ERROR.value:
            os._exit(1)
        time.sleep(STANDALONE_POLL_INTERVAL)


if __name__ == '__main__':
    if is_standalone():
        run_standalone()
    else:
        app.register()
        if is_native():
            app.handle_setup(client_id='1', coordinator=True, clients=['1'])
        else:
            run_app()
