# main.py
# -------------
from bayesianNetworkTester import BayesianNetworkTester
from bayesianNetwork import BayesianNetwork
import unittest

class TestBayesianNetwork(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)

        # Model 1
        self.root_model01 = './testcases/model-query-1/'
        self.model01 = BayesianNetwork(self.root_model01 + 'model.txt')
        self.model01_tester = BayesianNetworkTester(self.root_model01 + 'model.txt')

        # Model 2
        self.root_model02 = './testcases/model-query-2/'
        self.model02 = BayesianNetwork(self.root_model02 + 'model.txt')
        self.model02_tester = BayesianNetworkTester(self.root_model02 + 'model.txt')

        # Model asia
        self.root_asia = './testcases/asia/'
        self.asia = BayesianNetwork(self.root_asia + 'model.txt')
        self.asia_tester = BayesianNetworkTester(self.root_asia + 'model.txt')

        # Model child
        self.root_child = './testcases/child/'
        self.child = BayesianNetwork(self.root_child + 'model.txt')
        self.child_tester = BayesianNetworkTester(self.root_child + 'model.txt')

        # Model quiz
        self.root_quiz = './testcases/quiz/'
        self.quiz = BayesianNetwork(self.root_quiz + 'model.txt')
        self.quiz_tester = BayesianNetworkTester(self.root_quiz + 'model.txt')

    def equals(self, a, b):
        epsilon = 10e-5
        return abs(a-b) < epsilon

    def test_Result_Model1_Testcase01(self):
        result = self.model01.exact_inference(self.root_model01 + 'query1.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query1.txt')

        self.assertTrue(self.equals(result, expect))
    
    def test_Result_Model1_Testcase02(self):
        result = self.model01.exact_inference(self.root_model01 + 'query2.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query2.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model1_Testcase03(self):
        result = self.model01.exact_inference(self.root_model01 + 'query3.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query3.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model1_Testcase04(self):
        result = self.model01.exact_inference(self.root_model01 + 'query4.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query4.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model1_Testcase05(self):
        result = self.model01.exact_inference(self.root_model01 + 'query5.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query5.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model1_Testcase06(self):
        result = self.model01.exact_inference(self.root_model01 + 'query6.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query6.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model1_Testcase07(self):
        result = self.model01.exact_inference(self.root_model01 + 'query7.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query7.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model1_Testcase08(self):
        result = self.model01.exact_inference(self.root_model01 + 'query8.txt')
        expect = self.model01_tester.exact_inference(self.root_model01 + 'query8.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model2_Testcase01(self):
        result = self.model02.exact_inference(self.root_model02 + 'query1.txt')
        expect = self.model02_tester.exact_inference(self.root_model02 + 'query1.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model2_Testcase02(self):
        result = self.model02.exact_inference(self.root_model02 + 'query2.txt')
        expect = self.model02_tester.exact_inference(self.root_model02 + 'query2.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Model2_Testcase03(self):
        result = self.model02.exact_inference(self.root_model02 + 'query3.txt')
        expect = self.model02_tester.exact_inference(self.root_model02 + 'query3.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Asia_Query01(self):
        result = self.asia.exact_inference(self.root_asia + 'query1.txt')
        expect = self.asia_tester.exact_inference(self.root_asia + 'query1.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Child_Query01(self):
        result = self.child.exact_inference(self.root_child + 'query1.txt')
        expect = self.child_tester.exact_inference(self.root_child + 'query1.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Quiz_Query01(self):
        result = self.quiz.exact_inference(self.root_quiz + 'query1.txt')
        expect = self.quiz_tester.exact_inference(self.root_quiz + 'query1.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Quiz_Query02(self):
        result = self.quiz.exact_inference(self.root_quiz + 'query2.txt')
        expect = self.quiz_tester.exact_inference(self.root_quiz + 'query2.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Quiz_Query03(self):
        result = self.quiz.exact_inference(self.root_quiz + 'query3.txt')
        expect = self.quiz_tester.exact_inference(self.root_quiz + 'query3.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Quiz_Query04(self):
        result = self.quiz.exact_inference(self.root_quiz + 'query4.txt')
        expect = self.quiz_tester.exact_inference(self.root_quiz + 'query4.txt')

        self.assertTrue(self.equals(result, expect))

    def test_Result_Quiz_Query05(self):
        result = self.quiz.exact_inference(self.root_quiz + 'query5.txt')
        expect = self.quiz_tester.exact_inference(self.root_quiz + 'query5.txt')
        
        self.assertTrue(self.equals(result, expect))

if __name__ == "__main__":
    unittest.main()