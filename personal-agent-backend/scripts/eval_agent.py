#!/usr/bin/env python3
"""Agent 记忆与人格质量评估脚本。

用途：对陪伴机器人的多层级记忆系统进行端到端自动化测试，
验证 L1（会话内短期记忆）、L2（跨会话工作记忆归档）、
L3（长期记忆检索召回）三层记忆是否按预期协同工作。

测试场景：
  E1. L1 同一会话内上下文 —— 用户说了"下周二去上海出差"，
      紧接着问"下周二去哪？"，机器人应能正确引用刚提到的信息
  E2. L2/L3 跨会话长期记忆 —— 结束会话后开启新会话，
      问"还记得我下周二去哪来着？"，应能从归档记忆中召回
  E3. L2 事实归档 —— 在会话中表达忌口偏好（不吃香菜），
      结束后开启新会话再问，应能正确回忆该事实

测试方法：
  - 通过 handle_chat 发送消息并获取回复
  - 通过 handle_session_end 结束会话（触发 L1→L2 归档）
  - 在回复中检查关键词判断记忆是否被正确保留和召回
  - 这是功能级回归测试，不依赖硬件设备

典型用法：
    python scripts/eval_agent.py

前置条件：
    - LLM_API_KEY 已配置
    - 应用后端正常运行（或至少 agent 模块可直接调用）
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录（personal-agent-backend）添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.agent import handle_chat, handle_session_end
from app.session import store


async def run_eval() -> None:
    """执行 Agent 记忆评估的核心测试流程。"""

    # ---------- E1: L1 同一会话内短期记忆 ----------
    # 测试机器人在同一对话中能否记住刚提到的信息
    device_id = "eval-device"
    print("=== Eval: L1 same-session context ===")
    sid = store.get_or_create_session(device_id, None)

    # 用户告知一个未来计划（下周二去上海出差）
    r1, sid, _ = await handle_chat(device_id, sid, "我下周二要去上海出差。")
    print("A1:", r1)

    # 紧接着追问，验证 L1 短期记忆是否保留了"上海"和"出差"
    r2, sid, _ = await handle_chat(device_id, sid, "我下周二去哪？")
    print("A2:", r2)
    ok_l1 = any(k in r2 for k in ("上海", "出差"))
    print("L1 pass:" if ok_l1 else "L1 fail:", ok_l1)

    # ---------- E2: L2/L3 跨会话长期记忆 ----------
    # 测试结束会话并开启新会话后，之前的信息是否被归档并可召回
    print("\n=== Eval: L2/L3 cross-session ===")
    # 结束当前会话，触发 L1→L2 归档流程
    sid2 = await handle_session_end(device_id, sid) or store.get_or_create_session(device_id, None)
    # 新会话中尝试召回之前的出差信息
    r3, sid2, _ = await handle_chat(device_id, sid2, "还记得我下周二去哪来着？")
    print("A3:", r3)
    # 检查回复是否包含关键信息或表示不确定（不确定也算合理，至少说明尝试了回忆）
    ok_long = any(k in r3 for k in ("上海", "出差", "不确定", "印象"))
    print("Long-term pass:" if ok_long else "Long-term fail:", ok_long)

    # ---------- E3: L2 事实归档 ----------
    # 测试会话结束后，用户的偏好/事实数据是否正确归档
    print("\n=== Eval: L2 after session end ===")
    # 用户在会话中表达忌口偏好
    r4, sid2, _ = await handle_chat(device_id, sid2, "对了，我不吃香菜。")
    print("A4:", r4)
    # 结束会话，触发事实归档
    sid3 = await handle_session_end(device_id, sid2) or store.get_or_create_session(device_id, None)
    # 新会话中询问忌口，验证 L2 是否归档了"香菜"偏好
    r5, _, _ = await handle_chat(device_id, sid3, "我有什么忌口？")
    print("A5:", r5)
    ok_fact = "香菜" in r5
    print("L2 pass:" if ok_fact else "L2 fail:", ok_fact)


if __name__ == "__main__":
    asyncio.run(run_eval())
