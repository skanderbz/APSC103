Units
    Date: YYYY/MM/DD HOUR:MIN:SEC AM/PM
    NH4: mg/L
    NOx: mg/L
    DO: mg/L
    Temp: *C
    pH: N/A
    Predicted N20: mg/L
-----------------------------------------------------
Setup: Run Commands in your respective Command Line
*Note - Python required: 
 pip install python3

    Windows:

    python3 -m venv venv

    .\venv\Scripts\Activate.ps1

    #Run this if an error occurs:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

    pip install -r requirements.txt

-----------------------------------------------------
    Unix CLI (linux/MacOS):

    python3 -m venv venv

    source venv/bin/activate

    pip install -r requirements.txt

-----------------------------------------------------
First Time setup: Ensure you are in parent directory (APSC103 if you didn't specify when cloning)
    #If pkl file has not been initialized
    python3 main.py 

-----------------------------------------------------
Running GUI: Assuming Parent directory
    Open the user_inputs.xlsx file (Word, sheets etc.)
    Input known parameters under respective column (Use unit conventions described above)
    Close the excel (Error will be thrown if open)
    run:

    python3 excel_predictor.py

    Wait for "✅ Predictions added to Excel successfully." Message

    Empty N2O columns will now be filled out!

