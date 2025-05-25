//THIS CLASS IS IRRELEVANT FOR THE TIME BEING 05/25
class SortedMap {
    constructor() {
      this.map = new Map(); // Initialize the map
    }
  
    // Insert or update a key-value pair
    insert(key, value) {
      this.map.set(key, value); // Add the key-value pair
      this.order(); // Sort the map after each insertion
    }

    addVal(key, newVal) {
        // Get the current value for the key, default to 0 if the key doesn't exist
        const currentVal = this.map.get(key) || 0;

        // Add the new value to the current value
        this.map.set(key, currentVal + newVal);

        // Re-sort the map to maintain order
        this.order();
    }
  
    // Sort the map by values (ascending order)
    order() {
      // Convert the map to an array of [key, value] pairs
      const entries = Array.from(this.map.entries());
  
      // Sort the entries array by the value (ascending order)
      entries.sort((a, b) => {
        const valueA = a[1];
        const valueB = b[1];
        return valueA - valueB; // Smaller values come first
      });

      
  
      // Clear the map and reinsert the sorted entries
      this.map.clear();
      for (const [key, value] of entries) {
        this.map.set(key, value);
      }
    }

    contains(key) {
        return this.map.has(key); // Returns true if the key exists, false otherwise
      }
  
    // Convert the map to an array
    toArray() {
      return Array.from(this.map.entries());
    }
  }

  module.exports = SortedMap;
