import pandas as pd
import json

# 语料库字典(Kimi生成）
medical_corpus = {
    "Atelectasis": "肺不张，表现为肺叶或肺段的不透明影，可能与多种原因相关，如手术后、感染或肿瘤压迫。",
    "Cardiomegaly": "心脏增大，可能是由于心脏负荷增加、心脏瓣膜疾病或心肌病变引起的。",
    "Consolidation": "肺部实变，通常表现为肺叶或肺段的密度增高影，常见于肺炎、肺结核或其他肺部感染。",
    "Edema": "肺水肿，表现为肺野内弥漫性或局限性的密度增高影，可能由心力衰竭、肾功能不全或药物反应引起。",
    "Effusion": "胸腔积液，表现为胸腔内液体积聚，可能由炎症、肿瘤或心力衰竭引起。",
    "Emphysema": "肺气肿，表现为肺野内过度的透亮影，常见于慢性阻塞性肺疾病。",
    "Fibrosis": "纤维化，表现为肺野内的线状或网状影，可能与慢性炎症、感染后改变或肿瘤相关。",
    "Hernia": "疝，表现为器官从其正常解剖位置突出，可能需要手术治疗。",
    "Infiltration": "浸润，表现为肺野内的斑片状或结节状影，可能与感染、肿瘤或炎症相关。",
    "Mass": "肿块，表现为肺野内的圆形或不规则形影，可能是良性或恶性。",
    "Nodule": "结节，表现为肺野内的圆形小影，可能是良性或恶性，需要进一步检查。",
    "Pleural Thickening": "胸膜增厚，表现为胸膜的局部或弥漫性增厚，可能与炎症、肿瘤或外伤相关。",
    "Pneumonia": "肺炎，表现为肺野内的斑片状或实变影，通常伴有症状如发热、咳嗽。",
    "Pneumothorax": "气胸，表现为肺组织与胸壁之间的气体积聚，可能导致呼吸困难。",
    "Pneumoperitoneum": "腹腔积气，表现为腹腔内的气体积聚，可能与胃肠道穿孔或感染相关。",
    "Pneumomediastinum": "纵隔积气，表现为纵隔内的气体积聚，可能与外伤、手术或感染相关。",
    "Subcutaneous Emphysema": "皮下气肿，表现为皮下组织的气体积聚，可能与外伤或感染相关。",
    "Tortuous Aorta": "主动脉迂曲，表现为主动脉走行异常，可能与动脉硬化或其他血管疾病相关。",
    "Calcification of the Aorta": "主动脉钙化，表现为主动脉壁的钙化影，可能与动脉硬化相关。",
    "No Finding": "未发现异常，X光片显示正常。"
}


def read_csv(file_path):
    return pd.read_csv(file_path)


# 生成字符串模板
def create_medical_report(row):
    report = "X光片分析结果："
    findings = []

    for condition, value in row.items():
        if condition != 'id' and condition != 'subj_id' and condition != 'No Finding' and value == 1:
            findings.append(medical_corpus.get(condition, "未知异常"))

    if findings:
        report += "发现以下异常：" + "".join(findings)
    else:
        report += medical_corpus["No Finding"]

    if row["No Finding"] == 0:
        report += "需要进一步的医学评估。"

    return report


# 转换特征, image_file参数用于指定图片文件夹,由于使用了apply函数，在这里设置
def transform_features(row, image_file=''):
    if image_file == '':
        image_file = row['id']
    else:
        image_file = image_file + '/' + row['id']
    return {
        "img": image_file,
        "prompt": "请你分析这张X光图片",
        "label": create_medical_report(row)
    }


# 输出
def output_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


def main():
    input_file = 'csv/miccai2023_nih-cxr-lt_labels_val.csv'  # CSV文件路径
    output_file = 'output.json'  # JSON文件名
    df = read_csv(input_file)
    output_data = df.apply(transform_features, axis=1).tolist()
    output_to_json(output_data, output_file)
    print(f"数据已写入到 {output_file}")


if __name__ == "__main__":
    main()
