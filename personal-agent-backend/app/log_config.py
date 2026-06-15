"""Uvicorn 日志配置 —— 仅保留 WARNING 级别。

本模块的角色：
  陪伴机器人的控制台日志输出由 app.monitor.AgentMonitor 统一管理，
  Uvicorn 自身的 HTTP 访问日志和启动信息会污染控制台，因此将其静音到 WARNING。
  这样控制台只显示结构化的对话轮次摘要和后台事件，便于开发者快速定位问题。

使用方式：
  uvicorn.run(..., log_config=LOG_CONFIG, access_log=False)
"""

# Uvicorn 日志字典配置，遵循 Python logging.config.dictConfig 格式
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # 保留已有的 logger，不强制覆盖
    "formatters": {
        "default": {
            # 仅显示级别和消息，去掉时间戳减小噪音
            "format": "%(levelname)s %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",  # 输出到标准输出
            "formatter": "default",
        },
    },
    "loggers": {
        # Uvicorn 主 logger：静音到 WARNING，不再输出 HTTP 请求日志
        "uvicorn": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        # access log 也静音（配合 uvicorn.run access_log=False 双保险）
        "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        # app 模块：INFO 级别，用于监控沉默话题、会话整理等关键事件
        "app": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
    # 根 logger 也设为 WARNING，防止其他库输出 INFO 噪音
    "root": {"handlers": ["default"], "level": "WARNING"},
}
