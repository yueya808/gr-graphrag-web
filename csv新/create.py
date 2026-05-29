import pandas as pd
from neo4j import GraphDatabase
import os

# --- 配置信息，请务必修改为你的实际值 ---
URI = "neo4j+s://d4a3dc3e.databases.neo4j.io"          # 从Aura控制台获取
USERNAME = "neo4j"                 # 你的Aura用户名
PASSWORD = "FlDrBK_PI3rLgg_x2PtBUnhbORPy-OZkCAWE2qRNr9A"               # 你的Aura密码
CSV_FILE_PATH = r"D:\大学资料\（机器人及人工智能大赛26）03-CRAIC2026比赛规则及附件\0502\csv新\数控机床故障诊断图谱说明.csv"  # 你的CSV文件路径
BATCH_SIZE = 5000                  # 性能调优参数，可保持5000
# ----------------------------------------

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def create_graph(tx, batch):
    # 使用MERGE，如果节点已存在则匹配，避免重复创建
    query = """
    UNWIND $batch AS row
    MERGE (source:Component {name: row.源节点})
    MERGE (target:Component {name: row.目标节点})
    MERGE (source)-[:RELATION {type: row.关系类型, doc: row.文档说明}]->(target)
    """
    tx.run(query, batch=batch)

def import_csv():
    # 读取CSV文件，假设表头为: 源节点,关系类型,目标节点,文档说明
    df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8')
    total_rows = len(df)
    print(f"总数据量: {total_rows} 行")
    
    # 分批导入
    for start in range(0, total_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_rows)
        batch_data = df.iloc[start:end].to_dict('records')
        with driver.session() as session:
            session.execute_write(create_graph, batch_data)
        print(f"已完成: {end}/{total_rows} 行")

    print("数据导入完成！")

if __name__ == "__main__":
    import_csv()
    driver.close()