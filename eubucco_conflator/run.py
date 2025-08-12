import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eubucco_conflator import app
from eubucco_conflator.state import State

if __name__ == '__main__':
	
	# Initialize the state with your custom input file
	State.init('data/places-sample.parquet', logger=print)
	app.start(display_cols=["name",
							"address",
							"house_number",
							"country",
							"normalized_phone",
							"raw_websites",
							])
	