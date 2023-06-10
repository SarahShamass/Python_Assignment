import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
from unittest import mock
from sqlalchemy import create_engine

from index import (
    TrainingData,
    IdealFunction,
    TestData,
    DataProcessor,
    DataVisualization
)

class TrainingDataTestCase(unittest.TestCase):
    """Unit tests for the TrainingData class."""

    @classmethod
    def setUpClass(cls):
        # Create an in-memory SQLite database for testing
        engine = create_engine('sqlite:///:memory:', echo=True)
        Session = sessionmaker(bind=engine)
        cls.session = Session()
        TrainingData.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        # Close the session and clean up the database
        cls.session.close()

    def test_training_data(self):
        # Create a new training data object
        training_data = TrainingData(id=1, x=0.5, y1=1.0, y2=2.0, y3=3.0, y4=4.0)

        # Add the training data to the session
        self.session.add(training_data)
        self.session.commit()

        # Retrieve the training data from the database
        result = self.session.query(TrainingData).filter_by(id=1).first()

        # Check if the retrieved training data matches the original data
        self.assertEqual(result.id, 1)
        self.assertEqual(result.x, 0.5)
        self.assertEqual(result.y1, 1.0)
        self.assertEqual(result.y2, 2.0)
        self.assertEqual(result.y3, 3.0)
        self.assertEqual(result.y4, 4.0)

class IdealFunctionTestCase(unittest.TestCase):
    """Unit tests for the IdealFunction class."""

    @classmethod
    def setUpClass(cls):
        # Create an in-memory SQLite database for testing
        engine = create_engine('sqlite:///:memory:', echo=True)
        Session = sessionmaker(bind=engine)
        cls.session = Session()
        IdealFunction.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        # Close the session and clean up the database
        cls.session.close()

    def test_ideal_function(self):
        # Create a new ideal function object
        ideal_function = IdealFunction(id=1, x=0.5, y1=1.0, y2=2.0, y3=3.0, y4=4.0, y5=5.0)

        # Add the ideal function to the session
        self.session.add(ideal_function)
        self.session.commit()

        # Retrieve the ideal function from the database
        result = self.session.query(IdealFunction).filter_by(id=1).first()

        # Check if the retrieved ideal function matches the original data
        self.assertEqual(result.id, 1)
        self.assertEqual(result.x, 0.5)
        self.assertEqual(result.y1, 1.0)
        self.assertEqual(result.y2, 2.0)
        self.assertEqual(result.y3, 3.0)
        self.assertEqual(result.y4, 4.0)
        self.assertEqual(result.y5, 5.0)

class TestDataTestCase(unittest.TestCase):
    """Unit tests for the TestData class."""

    @classmethod
    def setUpClass(cls):
        """Create an in-memory SQLite database for testing"""
        engine = create_engine('sqlite:///:memory:', echo=True)
        Session = sessionmaker(bind=engine)
        cls.session = Session()
        TestData.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        """Close the session and clean up the database"""
        cls.session.close()

    def test_test_data(self):
        """# Create a new test data object"""
        test_data = TestData(id=1, x=0.5, y=1.0, delta_y=0.1, ideal_function=2)

        # Add the test data to the session
        self.session.add(test_data)
        self.session.commit()

        # Retrieve the test data from the database
        result = self.session.query(TestData).filter_by(id=1).first()

        # Check if the retrieved test data matches the original data
        self.assertEqual(result.id, 1)
        self.assertEqual(result.x, 0.5)
        self.assertEqual(result.y, 1.0)
        self.assertEqual(result.delta_y, 0.1)
        self.assertEqual(result.ideal_function, 2)

class DataProcessorTestCase(unittest.TestCase):
    """Unit tests for the DataProcessor class."""
    def setUp(self):
        self.data_processor = DataProcessor()

    def tearDown(self):
        self.data_processor = None

    def test_load_training_dataset(self):
        self.data_processor.load_training_dataset()
        training_dataset = self.data_processor.training_dataset
        self.assertIsNotNone(training_dataset)
        self.assertTrue(len(training_dataset) > 0)

    def test_load_test_dataset(self):
        self.data_processor.load_test_dataset()
        test_dataset = self.data_processor.test_dataset
        self.assertIsNotNone(test_dataset)
        self.assertTrue(len(test_dataset) > 0)

    @mock.patch('index.create_engine')
    def test_create_database(self, mock_create_engine):
        self.data_processor.training_dataset = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
        self.data_processor.ideal_functions = pd.DataFrame({'x': [1.0], 'y1': [2.0]})
        self.data_processor.create_database()
        mock_create_engine.assert_called_once_with('sqlite:///data.db', echo=True)

    def test_create_session(self):
        session = self.data_processor.create_session()
        self.assertIsNotNone(session)
        self.assertIsInstance(session, Session)

    @mock.patch.object(DataProcessor, 'create_session')
    def test_match_test_data_to_ideal_functions(self, mock_create_session):
        # Mock the create_session method to return a mock session
        mock_session = mock.Mock(spec=Session)
        mock_create_session.return_value = mock_session

        # Set up test data and ideal functions
        self.data_processor.test_dataset = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        self.data_processor.ideal_functions = pd.DataFrame({'x': [1.0], 'y1': [2.0]})

        # Run the method
        self.data_processor.match_test_data_to_ideal_functions()

        # Assertions
        mock_create_session.assert_called_once()
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_load_ideal_functions(self):
        self.data_processor.load_ideal_functions()
        ideal_functions = self.data_processor.ideal_functions
        self.assertIsNotNone(ideal_functions)
        self.assertTrue(len(ideal_functions) > 0)

class TestDataVisualization(unittest.TestCase):
    """Unit tests for the DataVisualization class."""

    def setUp(self):
        """Set up the test case."""
        self.data_processor = DataProcessor()
        self.visualization = DataVisualization(self.data_processor)

    def test_plot_ideal_functions(self):
        """Test the plot_ideal_functions method."""
        self.visualization.plot_ideal_functions()

    def test_plot_training_data(self):
        """Test the plot_training_data method."""
        self.visualization.plot_training_data()

    def test_visualize_data(self):
        """Test the visualize_data method."""
        self.data_processor.test_dataset = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
        self.visualization.visualize_data()
            
if __name__ == '__main__':
    unittest.main()
