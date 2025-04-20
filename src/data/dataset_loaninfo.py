import asyncio

from .base import BaseDataset
from .node import CommonDataNode, PubNode, SubNode
from .source import DataSourceCSV
from .process.loaninfo.preprocess import *



async def myhook(node):
    print(node.node_name)


class DatasetLoaninfoRaw(BaseDataset):
    def __len__(self):
        """데이터셋의 크기 반환"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """인덱스로 데이터 항목 접근"""
        pass

    async def pipeline(self, csv_path, ratio, random_state):
        node_0 = DataSourceCSV(csv_path)
        node_1 = CommonDataNode(node_0, node_name='node_1', process=cleaning_loaninfo_1)
        node_1.add_post_hook(myhook)
        node_2 = CommonDataNode(node_1, node_name='node_2', process=encoding_loaninfo_1)
        node_2.add_post_hook(myhook)
        node_3 = CommonDataNode(node_2, node_name='node_3', process=typing_loaninfo_1)
        node_3.add_post_hook(myhook)
        node_4 = PubNode(node_3, node_name='node_4', process=split_loaninfo_1, process_kwargs={"ratio":ratio, "random_state":random_state})
        node_4.add_post_hook(myhook)
        node_5 = SubNode(node_4, node_name='node_5', key='X_train')
        node_5.add_post_hook(myhook)
        node_6 = SubNode(node_4, node_name='node_6', key='y_train')
        node_6.add_post_hook(myhook)
        node_7 = SubNode(node_4, node_name='node_7', key='X_val')
        node_7.add_post_hook(myhook)
        node_8 = SubNode(node_4, node_name='node_8', key='y_val')
        node_8.add_post_hook(myhook)
        node_9 = SubNode(node_4, node_name='node_9', key='X_test')
        node_9.add_post_hook(myhook)
        node_10 = SubNode(node_4, node_name='node_10', key='y_test')
        node_10.add_post_hook(myhook)
        node_11 = CommonDataNode(node_5, node_name='node_11', process=get_max_values)
        node_11.add_post_hook(myhook)
        node_12 = CommonDataNode(node_5, node_11, node_name='node_12', process=scaler_loaninfo_1)
        node_12.add_post_hook(myhook)
        node_13 = CommonDataNode(node_7, node_11, node_name='node_13', process=scaler_loaninfo_1)
        node_13.add_post_hook(myhook)
        node_14 = CommonDataNode(node_9, node_11, node_name='node_14', process=scaler_loaninfo_1)
        node_14.add_post_hook(myhook)

        results = await asyncio.gather(node_12.get_data(), node_6.get_data(), node_13.get_data(), node_8.get_data(), node_14.get_data(), node_10.get_data())
        return results



if __name__ == "__main__":
    async def main():
        params = {
            "csv_path":"/workspace/Storage/template_structured/Data/raw/train.csv",
            "ratio": (0.6, 0.2, 0.2),
            "random_state":42
            }
        data = DatasetLoaninfoRaw(params)
        result = await data.data  # Add await here
        print(result)  # Print the result, not the coroutine
    
    import asyncio
    asyncio.run(main())