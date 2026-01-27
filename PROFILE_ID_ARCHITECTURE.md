# Profile ID Architecture

## The Pattern

### Storage Layer (CosmosDB)
- **All data stored under BASE user ID** (without `profile_` prefix)
- Example: Glucose, treatments, insights stored as `userId: "05bf0083..."`
- Partition key: `/userId` using base ID

### API Layer
- **Frontend passes profile ID** with `profile_` prefix for child profiles
- **Backend strips prefix** using `get_data_user_id()` before database queries
- Self profiles: `profile_{userId}` → strip to `{userId}`
- Child profiles: Use as-is OR strip if prefixed

### Function Contract

All API endpoints MUST use this helper:

```python
def get_data_user_id(profile_id: str) -> str:
    """
    Strip profile_ prefix for data access.
    
    Data is stored under base user ID without prefix.
    Profile IDs like 'profile_05bf...' map to data under '05bf...'
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix
    return profile_id
```

### Files That MUST Use get_data_user_id()

**ALL endpoints that query user data:**
- ✅ `api/v1/glucose.py` - get_current, get_history
- ✅ `api/v1/treatments.py` - get_recent, create, update, delete
- ✅ `api/v1/calculations.py` - IOB, COB, POB, dose calculations
- ✅ `api/v1/predictions.py` - BG predictions, ISF
- ✅ `api/v1/training.py` - Model training, ISF/ICR/PIR learning
- ✅ `api/v1/websocket.py` - Real-time glucose updates
- ✅ `api/v1/datasources.py` - Gluroo sync configuration
- ✅ `api/v1/insights.py` - AI insights generation

### Files That Use current_user.id AS-IS

**For ownership/permissions (NOT data queries):**
- `api/v1/sharing.py` - Share ownership uses account ID
- `api/v1/profiles.py` - Profile management uses account ID
- `auth/routes.py` - Authentication uses account ID from JWT

## Testing

Run this to verify data access:
```bash
python3 find_data_userids.py
```

Expected output:
```
[05bf0083-5598-43a5-aa7f-bd70b1f1be57]
  ✓ Latest glucose: 209 mg/dL
  ✓ History: 100 readings
  ✓ Treatments: 67
```
