import { createContext, useContext, useEffect, useState } from 'react';

const SelectedDatasetContext = createContext(null);

export function SelectedDatasetProvider({ children }) {
  const [selected, setSelectedState] = useState(() => {
    const raw = localStorage.getItem('mlready_last_dataset');
    return raw ? JSON.parse(raw) : null;
  });

  useEffect(() => {
    if (selected) localStorage.setItem('mlready_last_dataset', JSON.stringify(selected));
  }, [selected]);

  function setSelected(dataset) {
    setSelectedState(dataset ? { id: dataset.id, filename: dataset.filename } : null);
  }

  return (
    <SelectedDatasetContext.Provider value={{ selected, setSelected }}>
      {children}
    </SelectedDatasetContext.Provider>
  );
}

export function useSelectedDataset() {
  return useContext(SelectedDatasetContext);
}
