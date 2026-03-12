def get_risk_level(confidence, days_past_due, missed_payments):
    if days_past_due >= 90 or missed_payments >= 6:
        return "High Risk"
    if days_past_due >= 45 or missed_payments >= 3:
        return "Medium Risk"
    if confidence >= 80:
        return "Low Risk"
    return "Medium Risk"

def get_recommended_action(risk_level, days_past_due, legal_action_taken):
    if risk_level == "Low Risk":
        return "Send soft reminder and monitor repayment behavior."
    if risk_level == "Medium Risk":
        return "Call customer and offer structured repayment or restructuring."
    if risk_level == "High Risk":
        if str(legal_action_taken).strip().lower() == "yes":
            return "Escalate to legal recovery workflow and senior collections unit."
        return "Assign intensive recovery follow-up and consider legal escalation."
    return "Monitor account."