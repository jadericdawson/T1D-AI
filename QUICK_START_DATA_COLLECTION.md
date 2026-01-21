# Quick Start: Collecting Data to Improve IOB/COB/POB Curves

## The Problem You Identified
The BG pressure shading shows **above** the line even when BG is falling. This happens because our hardcoded activity curves (insulin peaks at 75min, carbs at 45min) don't match your actual physiology.

## The Solution
Collect real data to learn YOUR personalized curves, then use those instead of hardcoded values.

## What Data To Collect

### Critical Data (Start Collecting NOW)
1. **Full BG timeseries after each treatment** (not just 3 checkpoints)
2. **What our current curves predict** at each time point
3. **BG velocity** (how fast BG is changing) - this IS the activity rate!

###Why BG Velocity Matters
- High insulin **activity** → BG falling fast → negative velocity
- High carb **activity** → BG rising fast → positive velocity
- The velocity curve IS the absorption activity curve!

## Quick Win: Add This To Your Current System

### Step 1: Enable High-Resolution Data Collection

Add to your treatment logging endpoint:

```python
from services.absorption_curve_data_collector import AbsorptionCurveDataCollector

# After logging a treatment, schedule data collection
@app.post("/api/v1/treatments")
async def log_treatment(treatment: TreatmentCreate):
    # ... existing treatment logging ...

    # Schedule data collection 4 hours later
    collector = AbsorptionCurveDataCollector(user_id)
    # This runs in background after 4 hours
    asyncio.create_task(
        collect_and_store_later(treatment.id, user_id)
    )
```

### Step 2: Store The Data

Create new Cosmos container:

```bash
az cosmosdb sql container create \
  --account-name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --database-name T1D-AI-DB \
  --name absorption_curve_data \
  --partition-key-path /userId
```

### Step 3: Analyze Your Data (After 1-2 Weeks)

Once you have 20+ treatments collected:

```python
from services.absorption_curve_data_collector import AbsorptionCurveDataCollector

# Load your collected data
datasets = await load_absorption_datasets(user_id)

# Analyze: When does YOUR insulin actually peak?
insulin_peaks = []
for dataset in datasets:
    if dataset.insulinDose and dataset.actualPeakMin:
        insulin_peaks.append(dataset.actualPeakMin)

avg_peak = sum(insulin_peaks) / len(insulin_peaks)
print(f"Your insulin actually peaks at {avg_peak:.1f} minutes (theory says 75)")

# Analyze: When do YOUR carbs actually peak?
carb_peaks = []
for dataset in datasets:
    if dataset.carbs and dataset.actualPeakMin:
        carb_peaks.append(dataset.actualPeakMin)

avg_peak = sum(carb_peaks) / len(carb_peaks)
print(f"Your carbs actually peak at {avg_peak:.1f} minutes (theory says 45)")
```

## What You'll Learn

### Example: If You Discover
```
Your insulin peaks at 90 min (not 75)
Your carbs peak at 60 min (not 45)
```

### Then Update The Curves
```python
# In iob_cob_service.py
# OLD:
insulin_activity = insulin_activity_curve(time, peak_min=75, dia_min=240)
carb_activity = carb_activity_curve(time, peak_min=45, duration_min=180)

# NEW (personalized):
insulin_activity = insulin_activity_curve(time, peak_min=90, dia_min=240)  # YOUR peak
carb_activity = carb_activity_curve(time, peak_min=60, duration_min=180)  # YOUR peak
```

## Immediate Action Items

### Today (30 minutes)
- [x] Read ML_DATA_COLLECTION_REQUIREMENTS.md
- [ ] Add `absorption_curve_data_collector.py` to your imports
- [ ] Create `absorption_curve_data` Cosmos container

### This Week (2 hours)
- [ ] Add data collection trigger to treatment endpoint
- [ ] Test with 1-2 treatments
- [ ] Verify data is being stored correctly

### Next 2 Weeks (Passive)
- [ ] Let system collect 20+ treatment responses
- [ ] No action needed - just use the app normally

### Week 3 (1 hour)
- [ ] Run analysis script to find YOUR peak times
- [ ] Update curve parameters with YOUR values
- [ ] Deploy and see if pressure shading is more accurate!

## Expected Improvements

After using YOUR learned curves:

### Before (Hardcoded)
```
Insulin peaks at 75min for everyone
Carbs peak at 45min for everyone
→ Pressure might not match reality
→ Shading sometimes wrong direction
```

### After (Learned)
```
Insulin peaks at YOUR actual time (maybe 90min)
Carbs peak at YOUR actual time (maybe 60min)
→ Pressure reflects YOUR actual physiology
→ Shading should match BG direction
→ Leading indicator works!
```

## Key Insight

The BG velocity curve literally IS the activity curve you need!

If you plot:
- BG velocity after insulin dose → That's your insulin activity curve
- BG velocity after carb intake → That's your carb activity curve

Example:
```
Time: 0   15  30  45  60  75  90  105 120
BG:   150 150 145 140 132 125 120 118 116
Slope:  0  -0.3 -0.5 -1.0 -1.2 -1.0 -0.6 -0.3 -0.2

The slope curve IS your insulin activity!
Peak slope at 75min = peak insulin activity at 75min
```

## Files Created For You

1. **ML_DATA_COLLECTION_REQUIREMENTS.md** - Complete requirements
2. **absorption_curve_data_collector.py** - Ready-to-use data collector
3. **This file** - Quick start guide

## Next Steps

1. Start collecting the data (add collector to your API)
2. Wait 2 weeks for 20+ treatments
3. Analyze to find YOUR peak times
4. Update curve parameters
5. Enjoy accurate pressure shading! 🎉

## Questions This Will Answer

- "Why does pressure show above when I'm falling?" → Your curves don't match reality yet
- "When does MY insulin actually peak?" → Will know after 20 doses
- "Are my carbs really absorbed in 3 hours?" → Data will tell you
- "Does exercise change my curves?" → Can analyze exercise vs non-exercise data
- "Do I absorb faster in the morning?" → Can compare by time of day

## The Goal

Replace this:
```python
# Hardcoded guess
insulin_activity = curve(time, peak_min=75)
```

With this:
```python
# YOUR actual physiology
insulin_activity = learned_curve(time, user_id=your_id)
```

Then pressure shading will reflect YOUR body, not theory! 🎯
