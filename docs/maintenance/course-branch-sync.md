# 课程分支同步说明

## 首次整合记录

以下来源分支已按顺序合入 `lecture`，并分别保留来源历史：

| 来源分支 | 整合前 tip | `lecture` 合并提交 |
| --- | --- | --- |
| `SuperMimic` | `fe3004d5985e1318b5a9454ad1c25b7c1294783f` | `c930638642c5ebbaf2823e7958bc36d24992f9b0` |
| `SuperMimicPoleWeave` | `fa16a960fcaea4df7f537dc71354b4d4e029fa11` | `39ae0c4f67c2b3bdb3dca9b87f5ef7a257950100` |
| `MAPPO4V4` | `646197392ac41ebf97e067c9574d4db0dcd6e8ad` | `a40091f9cda3747399556b0882184def1689f213` |

前两个来源分支与初始 `lecture` 没有共同合并基点；`SuperMimicPoleWeave` 是从 `SuperMimic` 继续开发的。首次整合时先合入 `SuperMimic`，再合入绕杆分支，最后合入有共同历史的 4v4 分支。

## 后续同步

来源分支继续产生新提交后，在 `lecture` 上按以下顺序逐个合并，并为每个来源保留独立合并提交：

1. 合并 `SuperMimic` 的新提交。
2. 合并 `SuperMimicPoleWeave` 的新提交。
3. 合并 `MAPPO4V4` 的新提交。

每次合并前记录来源分支 tip；合并后用 `git merge-base --is-ancestor <来源提交> HEAD` 确认历史已进入当前分支。课程目录重组后，如果来源分支仍在旧路径上修改文件，需检查 Git 的重命名识别结果，并人工解决修改/删除冲突。不要通过重写已共享的来源分支历史来规避冲突。

来源分支名仅用于维护者同步和追溯，不作为学生可见的课程名称。
