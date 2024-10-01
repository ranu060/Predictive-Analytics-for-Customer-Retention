import React, { useState } from 'react';
import './App.css';

const App = () => {
  // State to store user inputs
  const [userData, setUserData] = useState({
    creditScore: '',
    age: '',
    gender: '',
    tenure: '',
    avgBalance: '',
    numProducts: '',
    isActiveMember: ''
  });

  // State to display results
  const [result, setResult] = useState(null);

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUserData({
      ...userData,
      [name]: value,
    });
  };

  // Mock function to calculate likelihood based on inputs
  const calculateLikelihood = () => {
    const { creditScore, age, tenure, avgBalance, numProducts, isActiveMember } = userData;
    // This is a placeholder logic, replace it with actual model or API call.
    const likelihood = {
      "6 months": (Math.random() * 100).toFixed(2),
      "12 months": (Math.random() * 100).toFixed(2),
      "3 years": (Math.random() * 100).toFixed(2),
      "5 years": (Math.random() * 100).toFixed(2),
    };

    setResult(likelihood);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Bank Customer Retention Predictor</h1>
        <form className="form">
          <div className="form-group">
            <label>Credit Score:</label>
            <input
              type="number"
              name="creditScore"
              value={userData.creditScore}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Age:</label>
            <input
              type="number"
              name="age"
              value={userData.age}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Gender:</label>
            <select name="gender" value={userData.gender} onChange={handleInputChange} required>
              <option value="">Select</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          <div className="form-group">
            <label>How long have you been with the bank? (in years):</label>
            <input
              type="number"
              name="tenure"
              value={userData.tenure}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Average Balance ($):</label>
            <input
              type="number"
              name="avgBalance"
              value={userData.avgBalance}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Number of Products from Bank:</label>
            <input
              type="number"
              name="numProducts"
              value={userData.numProducts}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Are you an active member of the bank?</label>
            <select
              name="isActiveMember"
              value={userData.isActiveMember}
              onChange={handleInputChange}
              required
            >
              <option value="">Select</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          <button
            type="button"
            className="submit-button"
            onClick={calculateLikelihood}
          >
            Calculate Likelihood
          </button>
        </form>

        {/* Display the results if they exist */}
        {result && (
          <div className="results">
            <h2>Likelihood of Leaving the Bank</h2>
            <ul>
              <li>Within 6 months: {result["6 months"]}%</li>
              <li>Within 12 months: {result["12 months"]}%</li>
              <li>Within 3 years: {result["3 years"]}%</li>
              <li>Within 5 years: {result["5 years"]}%</li>
            </ul>
          </div>
        )}
      </header>
    </div>
  );
};

export default App;