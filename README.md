# Concrete Mix Optimizer

A web application for optimizing concrete mix designs based on material constraints and performance requirements.

## Setup and Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Start the web server:

```bash
python3 app.py
```

3. Open your browser and go to:

```
http://localhost:3000
```

## Using the Application

1. **Set Optimization Target**

   - Choose to optimize for either cost or CO2 emissions

2. **Add Performance Requirements**

   - Set minimum strength (MPa)
   - Set minimum density (kg/m³)
   - Set maximum air content (%)
   - Set slump requirements (mm)

3. **Add Material Constraints**

   - Set minimum/maximum amounts for specific materials

4. **Adjust Advanced Settings** (optional)

   - Modify number of samples and iterations for more precise results

5. **Run Optimization**
   - Click the "Run Optimization" button
   - View results in the right panel

## Results

The application will display:

- Total cost and CO2 emissions
- Complete list of materials and their amounts
- Predicted concrete performance properties

All values are reported per liter of concrete.

## Development

- `app.py` - Flask application with API endpoints
- `mixer.py` - Core optimization code
- `index.html` - User interface
- `requirements.txt` - Required Python packages
