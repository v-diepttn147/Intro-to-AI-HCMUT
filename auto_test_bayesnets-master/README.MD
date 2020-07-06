# Auto Test Exact Inference For Bayesnets
*Hiện thực bằng thư viện [pgmpy](https://pgmpy.org/) :) mục đích để test kết quả nếu đem nộp bayesianNetworkTester.py sẽ bị 0 vì đề  yêu cầu chỉ dùng numpy.*
## Run Testcase Linux:
1. copy file `bayesianNetwork.py` vào folder `auto_test_bayesnets`
2. insall requirements linux: `pip install -r requirements.txt`
3. run: `python main.py`

## Run Testcase Windows:
1. copy file `bayesianNetwork.py` vào folder `auto_test_bayesnets`
2. install pytorch: `pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html`
2. insall requirements linux: `pip install -r requirements_windows.txt`
3. run: `python main.py`

## Thêm Testcase
1. tạo model trong `__init__` của `TestBayesianNetwork`:
    ```python
        def __init__(self, methodName):
            super().__init__(methodName)

            self.root_model01 = './testcases/model-query-1/'
            self.model01 = BayesianNetwork(self.root_model01 + 'model.txt')
            self.model01_tester = BayesianNetworkTester(self.root_model01 + 'model.txt')
    ```
2. Thêm testcase:
    ```python
        def test_Result_Model1_Testcase01(self):
            result = self.model01.exact_inference(self.root_model01 + 'query1.txt')
            expect = self.model01_tester.exact_inference(self.root_model01 + 'query1.txt')

            self.assertTrue(self.equals(result, expect))
    ```

