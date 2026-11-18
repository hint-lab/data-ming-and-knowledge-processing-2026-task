# 明日方舟攻略练度分级与个性化阵容适配推荐系统

## 挑战任务与组员

- 挑战项目题目：明日方舟攻略练度分级与个性化阵容适配推荐系统
- 组员：白昀、陈锦华

## 项目功能

系统读取干员账号 CSV，在社区攻略数据中检索指定关卡，按干员持有情况与配置差距排序，并展示推荐依据。攻略的培养练度与阵容获取门槛分别呈现。普通 7-18 提供基于既有作业的阵容适配、操作预览与 MAA 作业 JSON 下载。

## 数据与方法

- 数据来自社区作业列表、公开游戏资料和用户在本机导入的账号 CSV。作业列表带有阵容和部分配置字段，完整动作按需获取。
- 培养练度采用规则分类，根据精英阶段、等级、技能、模组和潜能判档。缺少等级时使用明确标示的估算值。获取信息列出已核实的限定、联动和星级数量。
- 个性化排序优先展示满足已检查条件的攻略，再按缺员、配置差距和操作量等代价排序，热度用于同等情况下的排序。备选组通过深度优先搜索检查合法阵容。
- 7-18 替代候选结合岗位条件、物理或法术输出、生命值和防御比较。公开示例可复现 MAA 文件下载流程，通关效果需通过实战验证。

## 源码与提交版本

- [独立项目仓库](https://github.com/baiyun070502/arknights-strategy-recommender)
- [本次提交对应的源码版本](https://github.com/baiyun070502/arknights-strategy-recommender/commit/e928e4142fe267f56b35ee3dea9268efcbade97b)

## 依赖与运行方法

使用 Python 3.11。网页依赖见项目的 `攻略推荐/requirements.txt`。首次安装依赖、下载游戏资料及构建社区作业索引时需要联网。

```powershell
git clone https://github.com/baiyun070502/arknights-strategy-recommender.git
cd arknights-strategy-recommender
git checkout --detach e928e4142fe267f56b35ee3dea9268efcbade97b
python -m pip install -r "攻略推荐/requirements.txt"
python "攻略推荐/scripts/import_game_data.py"
python -m streamlit run "攻略推荐/app.py"
```

1. 打开终端给出的本地地址。进入「7-18 作业适配」，选择「使用公开示例」，按页面步骤进入替换规则和最终方案。可以查看阵容与操作并下载公开示例 JSON。
2. 如需演示通用的本地攻略检索，先在项目根目录执行 `python "攻略推荐/build_index.py"`。完成后在页面进入「攻略推荐」，输入关卡查看匹配结果。可继续使用公开示例，也可在本机导入自己的干员 CSV。

网页分档、检索与 7-18 适配由 Python 完成。MAA 用于导入作业和后续实战，运行网页与生成 JSON 时无需安装 MAA。
