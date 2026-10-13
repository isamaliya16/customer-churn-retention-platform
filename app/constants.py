"""Shared enumerations (single source of truth for schemas, models, ML and the CLI)."""

CONTRACT_TYPES = ("month-to-month", "one-year", "two-year")
CONTRACT_TO_RAW = {"month-to-month": "Month-to-month", "one-year": "One year", "two-year": "Two year"}
RAW_TO_CONTRACT = {v: k for k, v in CONTRACT_TO_RAW.items()}

RISK_LEVELS = ("Low", "Medium", "High")
ACTION_STATUSES = ("open", "in_progress", "done", "dismissed")
ACTION_OUTCOMES = ("retained", "lost")
ROLES = ("admin", "analyst")

# Allowed values for the categorical customer-profile / model-input fields (IBM Telco vocabulary).
GENDERS = ("Male", "Female")
YES_NO = ("Yes", "No")
MULTIPLE_LINES = ("Yes", "No", "No phone service")
INTERNET_SERVICES = ("DSL", "Fiber optic", "No")
ADDON_VALUES = ("Yes", "No", "No internet service")
PAYMENT_METHODS = ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")

# Customer column -> model feature name
PROFILE_TO_FEATURE = {
    "gender": "gender",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
}
PROFILE_CHOICES = {
    "gender": GENDERS,
    "partner": YES_NO,
    "dependents": YES_NO,
    "phone_service": YES_NO,
    "multiple_lines": MULTIPLE_LINES,
    "internet_service": INTERNET_SERVICES,
    "online_security": ADDON_VALUES,
    "online_backup": ADDON_VALUES,
    "device_protection": ADDON_VALUES,
    "tech_support": ADDON_VALUES,
    "streaming_tv": ADDON_VALUES,
    "streaming_movies": ADDON_VALUES,
    "paperless_billing": YES_NO,
    "payment_method": PAYMENT_METHODS,
}
