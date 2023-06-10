import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, Integer, String
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, export_png
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


Base = declarative_base()


class TrainingData(Base):
    """Represents the training data table in the SQLite database."""

    __tablename__ = 'training_data'
    id = Column(Float, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)


class IdealFunction(Base):
    """Represents the ideal functions table in the SQLite database."""

    __tablename__ = 'ideal_functions'
    id = Column(Float, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)
    y5= Column(Float)
    y6 = Column(Float)
    y7 = Column(Float)
    y8 = Column(Float)
    y9 = Column(Float)
    y10 = Column(Float)
    y11 = Column(Float)
    y12 = Column(Float)
    y13 = Column(Float)
    y14 = Column(Float)
    y15= Column(Float)
    y16 = Column(Float)
    y17 = Column(Float)
    y18 = Column(Float)
    y19 = Column(Float)
    y20 = Column(Float)
    y21 = Column(Float)
    y22 = Column(Float)
    y23 = Column(Float)
    y24 = Column(Float)
    y25= Column(Float)
    y26 = Column(Float)
    y27 = Column(Float)
    y28 = Column(Float)
    y29 = Column(Float)
    y30 = Column(Float)
    y31 = Column(Float)
    y32 = Column(Float)
    y33 = Column(Float)
    y34 = Column(Float)
    y35= Column(Float)
    y36 = Column(Float)
    y37 = Column(Float)
    y38 = Column(Float)
    y39 = Column(Float)
    y40 = Column(Float)
    y41 = Column(Float)
    y42 = Column(Float)
    y43 = Column(Float)
    y44 = Column(Float)
    y45= Column(Float)
    y46 = Column(Float)
    y47 = Column(Float)
    y48 = Column(Float)
    y49 = Column(Float)
    y50 = Column(Float)


class TestData(Base):
    """Represents the test data table in the SQLite database."""

    __tablename__ = 'test_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    delta_y = Column(Float, name='delta_y_column')  # Rename the column
    ideal_function = Column(Float)


class DataProcessor:
    """Processes and analyzes the data."""

    def __init__(self):
        self.training_dataset = None
        self.test_dataset = None
        self.ideal_functions = None
        self.db_url = 'sqlite:///data.db'
        self.engine = create_engine(self.db_url, echo=True)

    def load_training_dataset(self):
        """Load the training dataset from a CSV file."""
        filename = "train.csv"
        self.training_dataset = pd.read_csv(filename)

    def load_test_dataset(self):
        """Load the test dataset from a CSV file."""
        filename = "test.csv"
        self.test_dataset = pd.read_csv(filename)

    def create_database(self):
        """Create a SQLite database and initialize tables."""
        self.engine = create_engine('sqlite:///data.db', echo=True)
        with self.engine.begin() as connection:
            self.training_dataset.to_sql('training_data', con=connection, if_exists='replace', index=False)
            if self.ideal_functions is not None:
                self.ideal_functions.to_sql('ideal_functions', con=connection, if_exists='replace', index=False)
            test_data_table = """
                CREATE TABLE IF NOT EXISTS test_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    x FLOAT,
                    y FLOAT,
                    delta_y_column FLOAT,
                    ideal_function FLOAT
                )
            """
            connection.execute(text(test_data_table))
    
    def create_session(self):
        """Create a new session with the database."""
        Session = sessionmaker(bind=self.engine)
        return Session()
    
    def match_test_data_to_ideal_functions(self):
        """Match the test data to the closest ideal function."""
        test_data = self.test_dataset
        ideal_functions = self.ideal_functions

        # Create a session and connect to the database
        session = self.create_session()

        # Iterate over each test data point
        for _, row in test_data.iterrows():
            x_test = row['x']
            y_test = row['y']

            # Calculate the deviation between the test data and each ideal function
            deviations = []
            for _, ideal_row in ideal_functions.iterrows():
                x_ideal = ideal_row['x']
                y_ideal = ideal_row.drop(['x']).values
                deviation = np.abs(y_test - y_ideal)
                deviations.append(deviation)

            # Find the ideal function with the minimum deviation
            min_deviation = np.min(deviations)
            ideal_function_index = np.argmin(deviations)

            # Save the test data and matched ideal function to the database
            test_data_entry = TestData(x=x_test, y=y_test, delta_y=min_deviation, ideal_function=ideal_function_index + 1)
            session.add(test_data_entry)

        # Commit and close the session
        session.commit()
        session.close()

    def load_ideal_functions(self):
        """Load the ideal functions from a CSV file."""
        filename = "ideal.csv"
        self.ideal_functions = pd.read_csv(filename)

    def run(self):
        """Run the data processing and analysis."""
        # Load data into variables
        self.load_training_dataset()
        self.load_test_dataset()
        self.load_ideal_functions()

        # Create the SQLite database
        self.create_database()

        # Match the test data to the ideal functions
        self.match_test_data_to_ideal_functions()


class DataVisualization:
    """Handles data visualization using Bokeh, SQLAlchemy and matplotlib."""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.engine = create_engine(self.data_processor.db_url)

    def plot_training_data(self):
        """Code to plot training data using SQLAlchemy and Bokeh"""
        with self.engine.connect() as connection:
            query = text("SELECT * FROM training_data")
            training_data = connection.execute(query).fetchall()

        output_notebook()  # Uncomment this line if running in a Jupyter Notebook

        # Extract x and y values from training data
        x = [row.x for row in training_data]
        y1 = [row.y1 for row in training_data]
        y2 = [row.y2 for row in training_data]
        y3 = [row.y3 for row in training_data]
        y4 = [row.y4 for row in training_data]

        # Create a new figure
        p = figure(title="Training Data", x_axis_label='X', y_axis_label='Y')

        # Plot training functions
        p.line(x, y1, legend_label='Y1')
        p.line(x, y2, legend_label='Y2', line_color='red')
        p.line(x, y3, legend_label='Y3', line_color='green')
        p.line(x, y4, legend_label='Y4', line_color='purple')

        # Add legend
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # Save the plot as a PNG file
        output_file("training_data_plot.html")
        export_png(p, filename="training_data_plot.png")

        # Show the plot
        show(p)

    def plot_ideal_functions(self):
        """Code to plot ideal functions using SQLAlchemy and Bokeh"""
        with self.engine.connect() as connection:
            query = text("SELECT * FROM ideal_functions")
            ideal_functions = connection.execute(query).fetchall()

        output_notebook()  # Uncomment this line if running in a Jupyter Notebook

        # Extract x and y values from ideal functions
        x = [row.x for row in ideal_functions]
        ys = [[getattr(row, f"y{i}") for row in ideal_functions] for i in range(1, 51)]

        # Create a new figure
        p = figure(title="Ideal Functions", x_axis_label='X', y_axis_label='Y')

        # Plot ideal functions
        for i, y in enumerate(ys, start=1):
            p.line(x, y, legend_label=f'Y{i}')

        # Add legend
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # Save the plot as a PNG file
        output_file("ideal_data_plot.html")
        export_png(p, filename="ideal_data_plot.png")

        # Show the plot
        show(p)

    def plot_test_data(self):
        """Plot the test data and assigned ideal functions."""
        test_data = self.data_processor.test_dataset

        if test_data is None:
            raise ValueError("Test data is not available.")

        # Extract x and y values from test data
        x = test_data['x']
        y = test_data['y']

        # Create a new figure
        fig, ax = plt.subplots()

        # Plot test data
        ax.plot(x, y, label='Test Data')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Test Data')

        # Add legend
        ax.legend()

        # Save the figure as a PNG file
        plt.savefig("test_data_plot.png")

    def visualize_data(self):
        """Visualize the training data, ideal functions, and test data."""
        self.plot_training_data()
        self.plot_ideal_functions()
        self.plot_test_data()
        
if __name__ == "__main__":
    """Main entry point of the program."""

    # Create an instance of DataProcessor and run the program
    data_processor = DataProcessor()
    data_processor.run()

    # Create an instance of DataVisualization and visualize the data
    data_visualization = DataVisualization(data_processor)
    data_visualization.visualize_data()


