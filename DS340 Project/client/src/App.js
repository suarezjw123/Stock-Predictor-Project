import React, { useState, useEffect} from 'react'

// Testing to see that this code can be committed to GitHub!

function App() {
  
  const [data, setData] = useState([{}])
  
  useEffect(() => {
    fetch("/members").then( 
      res => res.json()
    ).then(
        data => {
          setData(data)
          console.log(data)
    }
  )
}, [])
  
  
  
  
  return (
    <div>

      {(typeof data.members === 'undefined') ? (
         <p>Loading...</p>
      ) : (
        data.members.map((member, i) => (
          <p key={i}>{member}</p>
        ))
      )}

    </div>
  )
}

export default App