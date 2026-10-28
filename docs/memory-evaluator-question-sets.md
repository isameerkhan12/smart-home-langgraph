# Memory Evaluator Question Sets

Use each set as a sequence. Clear the memory store before starting a set, ask Question 1, allow the workflow to write memory, then ask Questions 2 and 3.

Expected sequence:

1. Question 1: `irrelevant`
2. Question 2: `sufficient`
3. Question 3: `supportive`

The first question should create a reusable memory containing the actual result. A recipe containing only a calculation method is `guidance`, not a complete answer.

## Set 1: Washing Machine

### Question 1 - expected `irrelevant`

> What was the total energy consumption of the Washing_Machine during March 2014, in kWh?

Expected stored memory:

> The Washing_Machine consumed `<value>` kWh during March 2014.

### Question 2 - expected `sufficient`

> What was the total energy consumption of the Washing_Machine during March 2014, in kWh?

The memory should be classified as `complete` and the set verdict should be `sufficient`.

### Question 3 - expected `supportive`

> What was the total energy consumption of the Washing_Machine during March and April 2014 combined, in kWh?

The March value can come from memory, but April must be calculated from the dataset.

## Set 2: Fridge

### Question 1 - expected `irrelevant`

> What was the average power consumption of the Fridge during June 2014, in watts?

### Question 2 - expected `sufficient`

> What was the average power consumption of the Fridge during June 2014, in watts?

### Question 3 - expected `supportive`

> What was the average power consumption of the Fridge during June and July 2014 combined, in watts?

June should come from memory and July should require a fresh calculation.

## Set 3: Dishwasher

### Question 1 - expected `irrelevant`

> How many kilowatt-hours did the Dishwasher consume during December 2014?

### Question 2 - expected `sufficient`

> How many kilowatt-hours did the Dishwasher consume during December 2014?

### Question 3 - expected `supportive`

> How many kilowatt-hours did the Dishwasher consume during December 2014, and how does that compare with January 2015?

December should come from memory and January should require a new calculation.

## Set 4: Electric Heater

### Question 1 - expected `irrelevant`

> What was the maximum recorded power consumption of the Electric_Heater during January 2015, in watts?

### Question 2 - expected `sufficient`

> What was the maximum recorded power consumption of the Electric_Heater during January 2015, in watts?

### Question 3 - expected `supportive`

> What was the maximum recorded power consumption of the Electric_Heater during January and February 2015?

The January maximum should come from memory and February should require a fresh calculation.

## Set 5: Appliance Comparison

### Question 1 - expected `irrelevant`

> Which appliance consumed more energy during March 2014: the Fridge or the Washing_Machine?

### Question 2 - expected `sufficient`

> Which appliance consumed more energy during March 2014: the Fridge or the Washing_Machine?

### Question 3 - expected `supportive`

> Which appliance consumed more energy during March 2014: the Fridge or the Dishwasher?

The first comparison should be reusable. The third question requires the Dishwasher value, which should not be in the original memory.

## Set 6: Aggregate Consumption

### Question 1 - expected `irrelevant`

> What was the average Aggregate power consumption during July 2014, in watts?

### Question 2 - expected `sufficient`

> What was the average Aggregate power consumption during July 2014, in watts?

### Question 3 - expected `supportive`

> What was the average Aggregate power consumption during July and August 2014 combined, in watts?

July should come from memory and August should require fresh computation.

## Testing Notes

Do not phrase Question 2 with any of these words:

- `recalculate`
- `verify`
- `recompute`

The evaluator prompt explicitly prevents `sufficient` when the user asks for recalculation or verification.

Also check the per-memory evaluations, not only the global verdict:

- Question 1 should have no relevant memory.
- Question 2 should contain a `complete` memory.
- Question 3 should contain at least one `partial` memory, while the new part requires tools.
