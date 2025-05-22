from typing import Any, Dict
import asyncio
import json
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

#mcp server 测试
#npx @modelcontextprotocol/inspector node build/index.js


# Initialize FastMCP server for Game tools (SSE)
mcp = FastMCP("mcp-test-server")

# Mock data storage
MOCK_SEASONS_DATA = {
    "S20": {
        "season_number": "S20",
        "theme": "荣耀觉醒",
        "start_date": "2025-01-15",
        "end_date": "2025-03-15",
        "description": "S20赛季'荣耀觉醒'主题围绕着游戏角色在战场上觉醒隐藏能力的故事展开，引入了全新的角色技能系统和特殊战场模式。",
        "rewards": ["传说皮肤", "专属头像框", "限定表情包"],
        "new_features": ["觉醒技能系统", "团队战术模式", "排位赛改革"]
    },
    "S19": {
        "season_number": "S19",
        "theme": "巅峰对决",
        "start_date": "2024-11-01",
        "end_date": "2025-01-14",
        "description": "S19赛季专注于竞技平衡性，优化了排位系统和匹配机制。",
        "rewards": ["史诗皮肤", "赛季边框", "专属称号"],
        "new_features": ["新匹配算法", "技能平衡调整", "观战系统升级"]
    }
}

MOCK_ACTIVITIES_DATA = [
    {
        "id": "CNY2025",
        "name": "2025龙年贺岁",
        "type": "seasonal",
        "start_date": "2025-02-01",
        "end_date": "2025-02-15",
        "description": "龙年春节特别活动，包含限定皮肤、游戏模式和丰厚奖励。",
        "rewards": ["限定龙年皮肤", "红包雨活动", "春节装饰"],
        "requirements": "完成每日任务获得活动币"
    },
    {
        "id": "LuckyDraw2025",
        "name": "新春好运抽奖",
        "type": "lottery",
        "start_date": "2025-02-01",
        "end_date": "2025-02-28",
        "description": "每日登录获得抽奖券，有机会获得稀有装备和游戏道具。",
        "rewards": ["稀有装备", "游戏道具", "专属头像"],
        "requirements": "每日登录获得1张抽奖券"
    },
    {
        "id": "Jan2025Special",
        "name": "2025新年特别活动",
        "type": "seasonal",
        "start_date": "2025-01-01",
        "end_date": "2025-01-15",
        "description": "庆祝2025年到来的特别活动，完成任务获得专属奖励。",
        "rewards": ["新年称号", "烟花特效", "专属表情"],
        "requirements": "完成新年主题任务"
    },
    {
        "id": "ValentineEvent",
        "name": "情人节甜蜜活动",
        "type": "seasonal",
        "start_date": "2025-02-10",
        "end_date": "2025-02-20",
        "description": "情人节特别活动，双人组队获得额外奖励。",
        "rewards": ["情侣皮肤", "爱心特效", "甜蜜称号"],
        "requirements": "与好友组队完成对战"
    }
]


async def simulate_api_delay():
    """模拟API请求延迟"""
    await asyncio.sleep(0.3)


def format_season_info(season_data: Dict[str, Any]) -> str:
    """格式化赛季信息"""
    return f"""
赛季信息:
==========
赛季: {season_data['season_number']} - {season_data['theme']}
时间: {season_data['start_date']} 至 {season_data['end_date']}
描述: {season_data['description']}

主要奖励:
{chr(10).join(f"• {reward}" for reward in season_data['rewards'])}

新增功能:
{chr(10).join(f"• {feature}" for feature in season_data['new_features'])}
"""


def format_activity_info(activity: Dict[str, Any]) -> str:
    """格式化活动信息"""
    return f"""
活动名称: {activity['name']}
活动类型: {activity['type']}
活动时间: {activity['start_date']} 至 {activity['end_date']}
活动描述: {activity['description']}
参与要求: {activity['requirements']}
奖励内容: {', '.join(activity['rewards'])}
"""


@mcp.tool()
async def search_seasons_knowledge(season: str) -> str:
    """搜索游戏赛季相关知识
    
    Args:
        season: 赛季名称或编号 (例如: S20, S19)
    """
    await simulate_api_delay()
    
    # 标准化赛季输入
    season_key = season.upper()
    
    if not season_key.startswith('S'):
        season_key = "S20"
    
    if season_key in MOCK_SEASONS_DATA:
        season_data = MOCK_SEASONS_DATA[season_key]
        return format_season_info(season_data)
    else:
        available_seasons = list(MOCK_SEASONS_DATA.keys())
        return f"未找到赛季 '{season}' 的信息。\n可用赛季: {', '.join(available_seasons)}"


@mcp.tool()
async def activity_search(query: str) -> str:
    """根据用户query搜索游戏活动
    
    Args:
        query: 用户query
    """
    await simulate_api_delay()
    
    # 搜索匹配的活动
    matching_activities = []
    query_lower = "活动"
    
    for activity in MOCK_ACTIVITIES_DATA:
        # 检查活动名称、描述或奖励中是否包含查询关键词
        if (query_lower in activity['name'].lower() or 
            query_lower in activity['description'].lower() or
            any(query_lower in reward.lower() for reward in activity['rewards']) or
            query_lower in activity['type'].lower()):
            matching_activities.append(activity)
    
    if not matching_activities:
        return f"未找到与 '{query}' 相关的活动。"
    
    result = f"找到 {len(matching_activities)} 个相关活动:\n"
    result += "=" * 40 + "\n"
    
    for i, activity in enumerate(matching_activities, 1):
        result += f"\n{i}. {format_activity_info(activity)}"
        if i < len(matching_activities):
            result += "\n" + "-" * 40 + "\n"
    
    return result


@mcp.tool()
async def activity_get_by_time(start_time: str, end_time: str) -> str:
    """根据时间范围获取活动
    
    Args:
        start_time: 开始时间 (YYYY/MM/DD 或 YYYY-MM-DD)
        end_time: 结束时间 (YYYY/MM/DD 或 YYYY-MM-DD)
    """
    await simulate_api_delay()
    
    # 标准化时间格式
    start_date = start_time.replace('/', '-')
    end_date = end_time.replace('/', '-')
    
    # 查找时间范围内的活动
    activities_in_range = []
    
    for activity in MOCK_ACTIVITIES_DATA:
        activity_start = activity['start_date']
        activity_end = activity['end_date']
        
        # 简单的时间范围检查 (实际项目中应使用更精确的日期比较)
        if (activity_start <= end_date and activity_end >= start_date):
            activities_in_range.append(activity)
    
    if not activities_in_range:
        return f"在时间范围 {start_time} 至 {end_time} 内未找到活动。"
    
    result = f"时间范围 {start_time} 至 {end_time} 内的活动 ({len(activities_in_range)} 个):\n"
    result += "=" * 50 + "\n"
    
    for i, activity in enumerate(activities_in_range, 1):
        result += f"\n{i}. {format_activity_info(activity)}"
        if i < len(activities_in_range):
            result += "\n" + "-" * 50 + "\n"
    
    return result

'''
@mcp.tool()
async def get_current_season() -> str:
    """获取当前赛季信息"""
    await simulate_api_delay()
    
    # 假设当前是S20赛季
    current_season = MOCK_SEASONS_DATA["S20"]
    result = "当前赛季信息:\n"
    result += format_season_info(current_season)
    return result


@mcp.tool()
async def get_active_activities() -> str:
    """获取当前正在进行的活动"""
    await simulate_api_delay()
    
    # 模拟当前时间为2025年2月，返回相应的活动
    current_activities = [
        activity for activity in MOCK_ACTIVITIES_DATA 
        if "2025-02" in activity['start_date'] or "2025-02" in activity['end_date']
    ]
    
    if not current_activities:
        return "当前没有正在进行的活动。"
    
    result = f"当前正在进行的活动 ({len(current_activities)} 个):\n"
    result += "=" * 40 + "\n"
    
    for i, activity in enumerate(current_activities, 1):
        result += f"\n{i}. {format_activity_info(activity)}"
        if i < len(current_activities):
            result += "\n" + "-" * 40 + "\n"
    
    return result
'''

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse
    
    parser = argparse.ArgumentParser(description='Run Game Assistant MCP SSE-based server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"Starting Game Assistant MCP Server...")
    print(f"Available tools:")
    print(f"  - search_seasons_knowledge: 搜索游戏赛季相关知识")
    print(f"  - activity_search: 搜索游戏活动")
    print(f"  - activity_get_by_time: 根据时间范围获取活动")
    print(f"  - get_current_season: 获取当前赛季信息")
    print(f"  - get_active_activities: 获取当前正在进行的活动")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=args.debug)

    uvicorn.run(starlette_app, host=args.host, port=args.port)