# ML Data Collection Requirements for Activity Curve Learning

## Objective
Learn personalized IOB/COB/POB activity curves from actual BG response data to improve pressure visualization and predictions.

## Current Data Collection ✓
**Source:** `src/services/ml_data_collector.py` + `MLTrainingDataPoint` schema

Currently collecting:
- ✅ Treatment details (carbs, protein, fat, fiber, GI, insulin dose)
- ✅ BG checkpoints at +30, +60, +90 minutes
- ✅ Prediction errors at each checkpoint
- ✅ Contextual features (time of day, lunar phase, etc.)

## Additional Data Needed for Activity Curve Learning

### 1. **High-Resolution BG Timeseries**
**Why:** Activity curves are continuous - we need to see the full shape, not just 3 points

**Needed:**
- ✅ Already collecting: All CGM readings (every 5 minutes)
- ⚠️ **Need to link:** Associate treatment ID with full BG timeseries for 4 hours post-treatment
- ⚠️ **Need to store:** Complete BG response window per treatment

**Implementation:**
```python
class TreatmentBGResponse(BaseModel):
    treatmentId: str
    treatmentTime: datetime
    bgReadings: List[Tuple[datetime, float]]  # Every 5 min for 4 hours
    insulinDose: Optional[float]
    carbsGrams: Optional[float]
    proteinGrams: Optional[float]
```

### 2. **Calculated IOB/COB/POB Over Time**
**Why:** We need to know what the *theoretical* curves predict vs actual BG response

**Needed:**
- ⚠️ **Calculate and store:** IOB/COB/POB values every 15 minutes for 4 hours post-treatment
- ⚠️ **Store activity levels:** Insulin/carb/protein activity (0-1) at each time point
- ⚠️ **Store predicted BG:** What our current curves predict at each time

**Implementation:**
```python
class AbsorptionCurveSnapshot(BaseModel):
    treatmentId: str
    minutesSinceTreatment: int
    theoreticalIOB: float
    theoreticalCOB: float
    theoreticalPOB: float
    insulinActivity: float  # 0-1 from activity curve
    carbActivity: float     # 0-1 from activity curve
    proteinActivity: float  # 0-1 from activity curve
    predictedBG: float
    actualBG: float
```

### 3. **BG Velocity (Rate of Change)**
**Why:** Activity is the *derivative* - how fast BG is changing tells us absorption rate

**Needed:**
- ⚠️ **Calculate and store:** BG slope (mg/dL per minute) at each timestamp
- ⚠️ **Store acceleration:** Second derivative to detect peaks

**Implementation:**
```python
# Add to TreatmentBGResponse
bgSlope: List[float]  # mg/dL per minute at each reading
bgAcceleration: List[float]  # Change in slope
```

### 4. **Isolated Treatment Windows**
**Why:** Overlapping treatments make it impossible to learn individual curves

**Current:**
- ✅ ml_data_collector.py filters for "clean" meals (no overlaps)

**Needed:**
- ✅ Keep filtering overlapping treatments
- ⚠️ **Add flag:** Mark treatments as "clean" vs "overlapped" in database
- ⚠️ **Track overlap distance:** How many minutes until next treatment?

### 5. **Treatment-Specific Features**
**Why:** Absorption varies by type, context, and individual factors

**Needed:**
- ⚠️ **Insulin type:** Rapid (Novolog/Humalog) vs Ultra-rapid (Fiasp/Lyumjev) vs Regular
- ⚠️ **Injection site:** Abdomen (fast) vs thigh (slow) vs arm
- ⚠️ **Meal composition ratios:** Carb:Protein:Fat ratio affects absorption
- ⚠️ **Exercise context:** Exercise within 2 hours before/after affects absorption
- ⚠️ **Stress markers:** If available (HRV, etc.)

### 6. **Ground Truth Labels**
**Why:** We need to know when absorption *actually* peaked

**Needed:**
- ⚠️ **Detect BG inflection points:** When BG stops rising/falling = absorption peaked
- ⚠️ **Measure actual onset:** Time from treatment to first BG movement
- ⚠️ **Measure actual peak:** Time from treatment to maximum BG rate-of-change

**Implementation:**
```python
class LearnedAbsorptionTiming(BaseModel):
    treatmentId: str
    actualOnsetMin: float  # Detected from BG slope turning positive/negative
    actualPeakMin: float   # Time when |BG slope| was maximum
    actualHalfLifeMin: float  # Time when effect dropped to 50%
    confidence: float  # 0-1, based on data quality
```

## Data Quality Requirements

### Minimum Requirements for Training
- ✅ At least 10 isolated insulin doses per user
- ✅ At least 10 isolated meals per food type
- ⚠️ **Need:** 20+ protein-only treatments per user
- ⚠️ **Need:** Full 4-hour CGM coverage (no gaps)
- ⚠️ **Need:** No sensor errors during window

### Data Validation
- ⚠️ **Flag invalid readings:** Sensor errors, compression lows
- ⚠️ **Filter outliers:** BG changes >10 mg/dL per 5 min (except post-treatment)
- ⚠️ **Require stability:** Pre-treatment BG should be stable (±10 mg/dL for 20 min)

## Current vs Needed Data Storage

### Current Schema (MLTrainingDataPoint)
```
✅ Treatment details
✅ 3 BG checkpoints (+30, +60, +90)
✅ Contextual features
❌ Full BG timeseries (only sampling 3 points)
❌ IOB/COB/POB over time
❌ BG velocity
❌ Activity curve values
```

### Needed: New Schema
```python
class AbsorptionLearningDataset(BaseModel):
    """Complete dataset for learning one treatment's absorption curve"""
    treatmentId: str
    userId: str
    treatmentTime: datetime

    # Treatment details
    insulinDose: Optional[float]
    insulinType: str
    injectionSite: str
    carbs: Optional[float]
    protein: Optional[float]
    fat: Optional[float]
    glycemicIndex: int

    # High-resolution timeseries (every 5 min for 4 hours)
    timestamps: List[datetime]  # 48 points
    actualBG: List[float]
    bgSlope: List[float]
    theoreticalIOB: List[float]
    theoreticalCOB: List[float]
    theoreticalPOB: List[float]
    insulinActivity: List[float]  # From current curve
    carbActivity: List[float]
    proteinActivity: List[float]
    predictedBG: List[float]  # What current curves predict

    # Derived labels (ground truth)
    actualOnsetMin: float
    actualPeakMin: float
    actualHalfLifeMin: float

    # Quality flags
    isClean: bool  # No overlapping treatments
    hasFullCoverage: bool  # No CGM gaps
    preTreatmentStable: bool  # BG was stable before
```

## Implementation Priority

### Phase 1: Enhance Current Collection (Week 1)
1. ✅ Already have: Basic ML data collector
2. ⚠️ **Add:** Store full BG timeseries per treatment
3. ⚠️ **Add:** Calculate and store theoretical IOB/COB/POB curves
4. ⚠️ **Add:** Calculate BG velocity at each point

### Phase 2: Feature Engineering (Week 2)
1. ⚠️ **Build:** Ground truth label extraction (onset, peak, half-life detection)
2. ⚠️ **Build:** Data quality filters
3. ⚠️ **Build:** Isolated treatment window detector

### Phase 3: Model Training Pipeline (Week 3)
1. ⚠️ **Build:** Activity curve learner training script
2. ⚠️ **Build:** Per-user curve personalization
3. ⚠️ **Build:** Curve validation against test data

### Phase 4: Production Integration (Week 4)
1. ⚠️ **Replace:** Hardcoded curves with learned curves
2. ⚠️ **Build:** Confidence scoring (use learned vs hardcoded)
3. ⚠️ **Build:** Continuous learning (retrain periodically)

## Key Metrics to Track

### Data Collection Metrics
- Clean treatments collected per day
- CGM coverage % (should be >95%)
- Treatment isolation rate (% with no overlaps)

### Model Performance Metrics
- MAE: Predicted vs actual BG at each time
- Curve shape error: How well curve fits actual response
- Onset timing error: Predicted vs detected onset
- Peak timing error: Predicted vs detected peak

### Production Metrics
- Pressure visualization accuracy: % where pressure direction matches BG direction
- User confidence: Survey data on whether predictions feel right
- Prediction improvement: MAE before vs after using learned curves

## Action Items for Better Data Collection

### Immediate (This Week)
1. ☐ Modify `ml_data_collector.py` to store full BG timeseries
2. ☐ Add IOB/COB/POB curve snapshots to training data
3. ☐ Calculate and store BG velocity

### Short Term (Next 2 Weeks)
1. ☐ Build ground truth label extraction (onset/peak detection)
2. ☐ Add data quality validation
3. ☐ Create `AbsorptionLearningDataset` schema

### Medium Term (Next Month)
1. ☐ Train initial absorption curve models
2. ☐ Validate learned curves against test data
3. ☐ Build confidence scoring system

### Long Term (2-3 Months)
1. ☐ Replace hardcoded curves with learned curves in production
2. ☐ Build continuous learning pipeline
3. ☐ Add per-food absorption curve learning

## Questions to Answer with Data

1. **How does insulin absorption vary by time of day?**
   - Collect: Timestamp, insulin dose, BG response
   - Learn: Circadian-adjusted activity curves

2. **How do meals with different macros absorb?**
   - Collect: Carb/protein/fat ratios, GI, BG response
   - Learn: Composition-specific absorption curves

3. **How does exercise affect absorption?**
   - Collect: Exercise timing/intensity, treatment response
   - Learn: Exercise-adjusted curves

4. **How stable are curves week-to-week?**
   - Collect: Same treatment repeated over time
   - Measure: Curve drift, retrain frequency needed

5. **Can we predict pressure BEFORE it happens?**
   - Collect: BG pressure (calculated), actual BG movement
   - Learn: Lead time between pressure and BG change
