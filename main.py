import os
import logging
from engines import EngineSimple

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL)



def main():
    engine = EngineSimple()
    engine.run()
    engine.plot()


if __name__ == "__main__":
    main()
